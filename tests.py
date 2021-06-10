import numpy as np
from run_sim import run_sim

all_tests_passed = True

def check(got, expected):
    global all_tests_passed
    if abs(got-expected) > 1e-10:
        print("!!!! Test Failed.", got, "is not the same as", expected)
        all_tests_passed = False
    else:
        print("Test Passed.")

print("---------------------- Checking installed packages ----------------------")
try:
    from hera_sim.visibilities.conversions import AzZaTransforms
    ok = True
except: ok = False
if ok:
    print("Testing hera_sim polybeam-astropy")
    uvdata, sim_time = run_sim("test_pyuvsim.yaml")

hera_sim_present = hera_sim_polybeam = hera_sim_polybeam_astropy = False
hera_gpu_present = hera_gpu_faster = False
healvis_present = healvis_updated = False
pyuvsim_present = False
try:
    import hera_sim
    hera_sim_present = True
    from hera_sim import beams
    hera_sim_polybeam = True
    from hera_sim.visibilities.conversions import AzZaTransforms
    hera_sim_polybeam_astropy = True
except: pass

try:
    import hera_gpu
    hera_gpu_present = True
    from hera_gpu import faster
    hera_gpu_faster = True
except: pass

try:
    import healvis
    healvis_present = True
    from healvis.version import updated
    healvis_updated = True
except: pass

try:
    import pyuvsim
    pyuvsim_present = True
except: pass

print("\n------------------------- Packages found --------------------------------\n")
if hera_sim_present:
    print("hera_sim is installed")
    if hera_sim_polybeam_astropy:
        print("\tpolybeam-astropy is installed")
    else:
        print("\tpolybeam-astropy is not installed")
        if hera_sim_polybeam:
            print("\tpolybeam is installed")
        else:
            print("\tpolybeam is not installed")
else:
    print("hera_sim is not installed")

if hera_gpu_present:
    print("hera_gpu is installed")
    if hera_gpu_faster:
        print("\tfaster gpu code is installed (this also fixes some accuracy issues)")
    else:
        print("\tfaster gpu code is not installed")
else:
    print("hera_gpu is not installed, or can't be imported due to CUDA missing")

if pyuvsim_present:
    print("pyuvsim is installed")
else:
    print("pyuvsim is not installed")

if healvis:
    print("healvis is installed")
    if healvis_updated:
        print("\tupdated healvis is installed (hugh's fork)")
    else:
        print("\tupdated healvis is not installed (hugh's fork)")
else:
    print("healvis is not installed")


print("\n------------------------- Starting tests --------------------------------\n")

if hera_sim_present:
    s = ">>>> Test hera_sim"
    if hera_sim_polybeam: s += " polybeam"
    elif hera_sim_polybeam_astropy: s += " polybeam-astropy"
    print(s)
    uvdata, sim_time = run_sim("test_hera_sim.yaml")
    print()
    check(abs(uvdata.get_data(0, 0, "XX")[0][0]), 0.0986073277611206)
    check(np.angle(uvdata.get_data(0, 1, "XX")[0][0]), -2.2464599726990793)

    if hera_sim_polybeam:
        uvdata, sim_time = run_sim("test_hera_sim_polybeam.yaml")
        print()
        check(abs(uvdata.get_data(0, 0, "XX")[0][0]), 0.46430402548254185)
        check(np.angle(uvdata.get_data(0, 1, "XX")[0][0]), -2.2464599726990793)

else: print(">>>> Skipping test hera_sim")

if hera_gpu_present:
    print(">>>> Test hera_gpu"+(" faster" if hera_gpu_faster else ""))
    uvdata, sim_time = run_sim("test_hera_gpu.yaml")
    print()
    if hera_gpu_faster:
        check(abs(uvdata.get_data(0, 0, "XX")[0][0]), 0.0986073277611206)
        check(np.angle(uvdata.get_data(0, 1, "XX")[0][0]), -2.2464599726990793)
    else:
        check(abs(uvdata.get_data(0, 0, "XX")[0][0]), 0.0986073277611206)
        check(np.angle(uvdata.get_data(0, 1, "XX")[0][0]), -2.8416234519478953)
else: print(">>>> Skipping test hera_gpu")

if pyuvsim_present:
    print(">>>> Test pyuvsim")
    uvdata, sim_time = run_sim("test_pyuvsim.yaml")
    print()
    check(abs(uvdata.get_data(0, 0, "XX")[0][0]), 0.0528842029860819)
    check(abs(uvdata.get_data(0, 1, "XX")[0][0]), 0.0528842029860819)
    check(np.angle(uvdata.get_data(0, 1, "XX")[0][0]), 1.9678439456133845)
else: print(">>>> Skipping test pyuvsim")

if healvis_present:
    if healvis_updated:
        print(">>>> Test healvis updated")
        vis, sim_time = run_sim("test_pyuvsim.yaml")
    else:
        print(">>>> Skipping test healvis (not implemented)")

else: print(">>>> Skipping test healvis")

if all_tests_passed:
    print()
    print("ALL TESTS PASSED.")
else:
    print()
    print("SOME TESTS FAILED.")
