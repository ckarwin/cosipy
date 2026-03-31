import pytest


pytest.importorskip("torch", reason="Optional torch dependencies not installed")

from cosipy.background_estimation.ml import ContinuumEstimationInterp, ContinuumEstimationNN
from cosipy import test_data

def test_continuum_background_estimation(tmp_path,monkeypatch):
   
    monkeypatch.chdir(tmp_path)

    instance = ContinuumEstimationNN() 
     
    # Test main method:
    data_yaml = test_data.path / "inputs_crab_continuum_bg_estimation_testing.yaml"
    input_data = test_data.path / "crab_bkg_binned_data_for_continuum_bg_testing.hdf5"
    psr_file = test_data.path / "test_precomputed_response.h5"
   
    instance.estimate_bg(input_data, psr_file, background_model=None,
            training_mode="self", containment=0.6, epochs=1, model_type="gcn",
            nn_model="new", nn_model_file=None, nn_model_savename="inpainting_nn_model",
            lr=1e-3, self_mask_fraction=0.1, lambda_sup=0.5, lambda_self=0.5,
            prefix="inpainted", visualize=False, em_bin=1, phi_bin=1,
            evaluate_only=False, inpainted_file=None,
            evaluate=False, show_plots=False)

    instance.estimate_bg(input_data, psr_file, background_model=input_data,
        training_mode="supervised", containment=0.6, epochs=1, em_bin=1, phi_bin=1,
        evaluate=True,visualize=True)

    instance.estimate_bg(input_data, psr_file, background_model=input_data,
        training_mode="supervised", containment=0.6, epochs=1, em_bin=1, phi_bin=1,
        nn_model="load", nn_model_file="inpainting_nn_model.pth", nn_model_savename="inpainting_nn_model_new")

    instance.estimate_bg(input_data, psr_file, background_model=input_data,
        training_mode="supervised", containment=0.6, epochs=1, em_bin=1, phi_bin=1,
        nn_model="load_full", nn_model_file="inpainting_nn_model.pth", nn_model_savename="inpainting_nn_model_new")

    instance.estimate_bg(input_data, psr_file, background_model=input_data,
        training_mode="hybrid", containment=0.6, epochs=1, em_bin=1, phi_bin=1)
    
    instance.plot_training_loss("inpainting_nn_model_training_loss.npy",1,"training_loss",show_plot=False)

    # Test simple inpainging method:
    instance = ContinuumEstimationInterp()
    instance.estimate_bg(input_data, psr_file)
