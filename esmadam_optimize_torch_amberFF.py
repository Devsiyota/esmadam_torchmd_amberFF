import os
import random

import esm
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn.utils as utils

from torchmd_amber_energy import TorchMDAmberEnergy
from torchmd_amber_energy import energy_to_float
from torchmd_amber_energy import print_energy_details

from hydrogens_template import (
    make_atom14_atomic_numbers,
    add_template_hydrogens_to_output_nn,
    extract_heavy_and_template_hydrogen_pos_z,
    write_pdb_with_template_hydrogens,
)




DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRECISION = torch.float32

ESMFOLD_CACHE_DIR = "/depot/chen4116/data/deven_esmfold_checkpoints"

SEQUENCE = "TTYKLILNLKQAKEEAIKELVDAGTAEKYFKLIANAKTVEGVWTYKDEIKTFTVTE"

# These two files must describe the same atom order/count.
AMBER_PRMTOP = "protein.prmtop"
PROTEIN_AMBER_PDB = "protein_amber.pdb"

NUM_STEPS = 100
LEARNING_RATE = 5e-3
LATENT_NOISE_SCALE = 0.1

# Do not use clash detection as a hard kill except for catastrophic collapse.
# Let the optimizer feel clashes through loss_clash.
CLASH_CUTOFF = 1.00
CLASH_WEIGHT = 500.0
CATASTROPHIC_MIN_DIST = 0.96

CA_WEIGHT = 10.0
MAX_LATENT_GRAD_NORM = 1.0

OUTPUT_DIR = "latent_accept_torch_amber_outputs_template_H_amber_order"
SEED = 42

# Same default AMBER/TorchMD terms as torchmd_amber_energy.py.
# Keep this here only so the optimization run is explicit/reproducible.
AMBER_TERMS = [
    "bonds",
    "angles",
    "dihedrals",
    "impropers",
    "1-4",
    "electrostatics",
    "lj",
]


# ============================================================
# Setup helpers
# ============================================================

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_esmfold_model():
    print("Loading ESMFold...")

    torch.hub.set_dir(ESMFOLD_CACHE_DIR)

    model_esm = esm.pretrained.esmfold_v1()
    model_esm = model_esm.eval().to(DEVICE)

    for p in model_esm.parameters():
        p.requires_grad_(False)

    #print("ESMFold loaded")
    return model_esm


def build_amber_energy_backend():
    #print("Building TorchMD AMBER energy backend...")

    amber_energy = TorchMDAmberEnergy(
        prmtop=AMBER_PRMTOP,
        pdb_file=PROTEIN_AMBER_PDB,
        device=DEVICE,
        precision=PRECISION,
        terms=AMBER_TERMS,
    )

    amber_energy.print_system_info()
    #print("TorchMD AMBER energy backend ready")

    return amber_energy


# ============================================================
# Geometry / diagnostics
# ============================================================

def compute_ca_distances(coords):
    diff_i1 = coords[:, :-1, :] - coords[:, 1:, :]
    dist_i1 = torch.norm(diff_i1, dim=-1)
    return dist_i1


def output_to_pdb_string(model_esm, output_nn):
    output_cpu = {}

    for k, v in output_nn.items():
        if torch.is_tensor(v):
            output_cpu[k] = v.detach().cpu()
        else:
            output_cpu[k] = v

    pdb_str = model_esm.output_to_pdb(output_cpu)[0]
    return pdb_str


def write_output_nn_pdb(model_esm, output_nn, filename):
    pdb_str = output_to_pdb_string(model_esm, output_nn)

    with open(filename, "w") as f:
        f.write(pdb_str)


def check_finite_scalar(x, name):
    if not torch.isfinite(x):
        #print(f"{name}:", x)
        raise RuntimeError(f"{name} is NaN or Inf")


def report_closest_contact(pos, z, name="pos"):
    with torch.no_grad():
        if not torch.isfinite(pos).all():
            #print(f"{name} contains NaN or Inf")
            return None

        d = torch.cdist(pos.detach(), pos.detach())
        eye = torch.eye(d.shape[0], dtype=torch.bool, device=d.device)
        d = d.masked_fill(eye, 1e6)

        min_dist = d.min()
        idx = torch.nonzero(d == min_dist, as_tuple=False)[0]

        i = idx[0].item()
        j = idx[1].item()

        #print(f"{name} minimum interatomic distance: {min_dist.item():.4f} Å")
        #print(f"{name} closest atom indices: [{i}, {j}]")
        #print(f"{name} closest atom z: {z[i].item()} {z[j].item()}")

        return min_dist.item()


def compute_clash_loss(pos, cutoff=1.00):
    d = torch.cdist(pos, pos)
    eye = torch.eye(d.shape[0], dtype=torch.bool, device=d.device)
    d = d.masked_fill(eye, 1e6)

    clash = F.relu(cutoff - d)
    loss_clash = torch.mean(clash ** 2)

    return loss_clash


def get_amber_ordered_pos_z_from_output(output_nn, sequence, z14):
    output_nn_H = add_template_hydrogens_to_output_nn(
        output_nn=output_nn,
        sequence=sequence,
        amber_pdb_path=PROTEIN_AMBER_PDB,
    )

    pos_all, z_all = extract_heavy_and_template_hydrogen_pos_z(
        output_nn_H=output_nn_H,
        z14=z14,
    )

    return output_nn_H, pos_all, z_all


def compute_amber_energy(amber_energy, pos_all):
    if pos_all.dim() != 2 or pos_all.shape[1] != 3:
        raise RuntimeError(f"Expected pos_all shape [natoms, 3], got {tuple(pos_all.shape)}")

    pos_amber = pos_all.unsqueeze(0).to(DEVICE)

    energy = amber_energy.energy(pos_amber)

    return energy


def run_initial_force_check(amber_energy, pos_all):
    #print()
    #print("Initial TorchMD AMBER energy/force check:")

    pos_amber = pos_all.detach().unsqueeze(0).clone().to(DEVICE)

    E_raw, force_buffer = amber_energy.raw_energy_and_force(pos_amber)

    print_energy_details(E_raw)
    #print("initial force tensor shape:", force_buffer.shape)
    #print("initial force norm:", float(torch.linalg.norm(force_buffer).detach().cpu()))
    #print("initial force min:", float(force_buffer.detach().cpu().min()))
    #print("initial force max:", float(force_buffer.detach().cpu().max()))

    if torch.isnan(force_buffer).any():
        raise RuntimeError("Initial AMBER forces contain NaN")

    if torch.isinf(force_buffer).any():
        raise RuntimeError("Initial AMBER forces contain Inf")

    test_pos = pos_amber.detach().clone().requires_grad_(True)
    test_energy = amber_energy.energy(test_pos)

    #print("test_energy:", float(test_energy.detach().cpu()))
    #print("test_energy requires_grad:", test_energy.requires_grad)
    #print("test_energy grad_fn:", test_energy.grad_fn)

    test_energy.backward()

    if test_pos.grad is None:
        raise RuntimeError("Custom AMBER backward did not create test_pos.grad")

    #print("test pos.grad norm:", float(torch.linalg.norm(test_pos.grad).detach().cpu()))

    return energy_to_float(E_raw)



def main():
    seed_everything(SEED)
    torch.autograd.set_detect_anomaly(True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_esm = load_esmfold_model()
    amber_energy = build_amber_energy_backend()

    natoms_from_prmtop = amber_energy.mol.numAtoms

    z14 = make_atom14_atomic_numbers(SEQUENCE, DEVICE)

    # ESMFold structure


    with torch.no_grad():
        output, esm_s_output = model_esm.infer(SEQUENCE)

    write_output_nn_pdb(
        model_esm=model_esm,
        output_nn=output,
        filename=os.path.join(OUTPUT_DIR, "esmfold_heavy.pdb"),
    )

    output_H, init_pos_all, init_z_all = get_amber_ordered_pos_z_from_output(
        output_nn=output,
        sequence=SEQUENCE,
        z14=z14,
    )

    #print("Initial all-atom natoms:", init_pos_all.shape[0])
    #print("Expected prmtop natoms:", natoms_from_prmtop)
    #print("Initial hydrogen_positions shape:", output_H["hydrogen_positions"].shape)
    #print("Initial hydrogen_atom_exists shape:", output_H["hydrogen_atom_exists"].shape)

    if init_pos_all.shape[0] != natoms_from_prmtop:
        raise RuntimeError(
            "Generated AMBER-ordered atom count does not match prmtop. "
            f"generated={init_pos_all.shape[0]}, prmtop={natoms_from_prmtop}. "
            "This must be fixed before optimization."
        )

    init_min_dist = report_closest_contact(
        pos=init_pos_all,
        z=init_z_all,
        name="initial_esmfold_with_template_H_amber_order",
    )

    #if init_min_dist is not None and init_min_dist < CATASTROPHIC_MIN_DIST:
        #raise RuntimeError(
            #f"Initial catastrophic atom collapse: min_dist={init_min_dist:.4f} Å"
        #)

    init_energy = run_initial_force_check(
        amber_energy=amber_energy,
        pos_all=init_pos_all,
    )

    print("Initial AMBER energy:", init_energy)

    write_pdb_with_template_hydrogens(
        output_nn_H=output_H,
        sequence=SEQUENCE,
        filename=os.path.join(OUTPUT_DIR, "esmfold_with_template_H_amber_order_debug.pdb"),
        amber_pdb_path=PROTEIN_AMBER_PDB,
        chain_id="",
    )

    # Latent optimization setup

    #initial_esm_s = esm_s_output.detach().clone().to(DEVICE)
    #initial_esm_s = initial_esm_s + LATENT_NOISE_SCALE * torch.randn_like(initial_esm_s)
    initial_esm_s = LATENT_NOISE_SCALE * torch.randn_like(esm_s_output)
    initial_esm_s = initial_esm_s.detach().to(DEVICE).requires_grad_(True)

    #print("esm_s_output device after initial infer:", esm_s_output.device)
    #print("optimized initial_esm_s device:", initial_esm_s.device)
    #print("first ESMFold parameter device:", next(model_esm.parameters()).device)

    if initial_esm_s.device != next(model_esm.parameters()).device:
        raise RuntimeError(
            f"Device mismatch before optimization: "
            f"initial_esm_s={initial_esm_s.device}, "
            f"model={next(model_esm.parameters()).device}"
        )

    optimizer = torch.optim.Adam(
        [initial_esm_s],
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
        verbose=True,
    )

    x = output["positions"]
    nbond = x.shape[2] - 1
    ca_dist_target = torch.ones((1, nbond), device=DEVICE) * 3.8

    best_loss = None
    best_output_nn_H = None
    best_step = None

    # Optimization loop

    #print()
    #print("Starting ESMFold latent optimization using TorchMD AMBER")
    #print("learning_rate:", LEARNING_RATE)
    #print("latent_noise_scale:", LATENT_NOISE_SCALE)
    #print("clash_cutoff:", CLASH_CUTOFF)
    #print("clash_weight:", CLASH_WEIGHT)
    #print("ca_weight:", CA_WEIGHT)
    #print("num_steps:", NUM_STEPS)
    #print()

    for step in range(NUM_STEPS):
        optimizer.zero_grad()

        add_term = initial_esm_s.to(DEVICE)

        # Keep esm_s_input on exactly the same device as the ESMFold model.
        output_nn, esm_s_nn = model_esm.infer(
            SEQUENCE,
            esm_s_input=add_term,
        )

        output_nn_H, pos_all, z_all = get_amber_ordered_pos_z_from_output(
            output_nn=output_nn,
            sequence=SEQUENCE,
            z14=z14,
        )

        if pos_all.shape[0] != natoms_from_prmtop:
            raise RuntimeError(
                f"Step {step}: generated atom count {pos_all.shape[0]} does not match prmtop {natoms_from_prmtop}"
            )

        min_dist = report_closest_contact(
            pos=pos_all,
            z=z_all,
            name=f"step_{step:04d}_pos_all",
        )

        #if min_dist is not None and min_dist < CATASTROPHIC_MIN_DIST:
            #raise RuntimeError(
                #f"Catastrophic atom collapse at step {step}: min_dist={min_dist:.4f} Å"
            #)

        energy = compute_amber_energy(
            amber_energy=amber_energy,
            pos_all=pos_all,
        )

        if not torch.isfinite(energy):
            #print("TorchMD AMBER returned non-finite energy")
            #print("energy:", energy)
            raise RuntimeError("Stopping because TorchMD AMBER returned NaN or Inf energy")

        cur_ca_dist = compute_ca_distances(
            output_nn["positions"][-1, :, :, 1, :]
        )

        loss_ca = F.mse_loss(ca_dist_target, cur_ca_dist)
        #loss_e = energy / pos_all.shape[0]
        loss_e = energy

        loss_clash = compute_clash_loss(
            pos=pos_all,
            cutoff=CLASH_CUTOFF,
        )

        loss = loss_e + CA_WEIGHT * loss_ca + CLASH_WEIGHT * loss_clash

        check_finite_scalar(loss_e, "loss_e")
        check_finite_scalar(loss_ca, "loss_ca")
        check_finite_scalar(loss_clash, "loss_clash")
        check_finite_scalar(loss, "loss")

        print(
            f"Step {step:04d} | "
            f"loss={loss.item():.6f} | "
            f"loss_e_per_atom={loss_e.item():.6f} | "
            f"raw_amber_energy={energy.item():.6f} | "
            f"loss_ca={loss_ca.item():.6f} | "
            f"loss_clash={loss_clash.item():.6f} | "
            f"min_dist={min_dist:.4f} | "
            f"natoms={pos_all.shape[0]}"
        )

        loss.backward()

        if initial_esm_s.grad is None:
            raise RuntimeError(
                "initial_esm_s.grad is None. Gradient is not reaching esm_s_input."
            )

        if not torch.isfinite(initial_esm_s.grad).all():
            bad_grad_count = (~torch.isfinite(initial_esm_s.grad)).sum().item()
            #print("Bad gradient count:", bad_grad_count)
            raise RuntimeError("initial_esm_s.grad contains NaN or Inf")

        grad_norm_before_clip = initial_esm_s.grad.norm().item()
        #print("initial_esm_s grad norm before clip:", grad_norm_before_clip)

        utils.clip_grad_norm_([initial_esm_s], max_norm=MAX_LATENT_GRAD_NORM)

        grad_norm_after_clip = initial_esm_s.grad.norm().item()
        #print("initial_esm_s grad norm after clip:", grad_norm_after_clip)

        current_loss = float(loss.detach().cpu())

        if best_loss is None or current_loss < best_loss:
            best_loss = current_loss
            best_step = step
            best_output_nn_H = output_nn_H

        optimizer.step()
        scheduler.step(current_loss)

        if step % 1 == 0 or step == NUM_STEPS - 1:
            heavy_pdb = os.path.join(OUTPUT_DIR, f"structure_{step:04d}_heavy.pdb")
            h_pdb = os.path.join(OUTPUT_DIR, f"structure_{step:04d}_template_H_amber_order.pdb")

            write_output_nn_pdb(
                model_esm=model_esm,
                output_nn=output_nn,
                filename=heavy_pdb,
            )

            write_pdb_with_template_hydrogens(
                output_nn_H=output_nn_H,
                sequence=SEQUENCE,
                filename=h_pdb,
                amber_pdb_path=PROTEIN_AMBER_PDB,
                chain_id="",
            )

            #print("Saved:", heavy_pdb)
            #print("Saved:", h_pdb)

    if best_output_nn_H is not None:
        best_pdb = os.path.join(OUTPUT_DIR, "best_template_H_amber_order.pdb")
        write_pdb_with_template_hydrogens(
            output_nn_H=best_output_nn_H,
            sequence=SEQUENCE,
            filename=best_pdb,
            amber_pdb_path=PROTEIN_AMBER_PDB,
            chain_id="",
        )
        print("Best step:", best_step)
        print("Best loss:", best_loss)
        print("Saved best:", best_pdb)

    print("DONE")


if __name__ == "__main__":
    main()
