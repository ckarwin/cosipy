# Imports
import sys
import numpy as np
import pandas as pd
import healpy as hp
import argparse
import matplotlib.pyplot as plt
from histpy import Histogram, Axes
from cosipy.background_estimation import ContinuumEstimation
from cosipy.response import FullDetectorResponse, DetectorResponse
from cosipy.interfaces import BinnedBackgroundInterface
from typing import Dict, Tuple, Union, Any, Type, Optional, Iterable
from astropy import units as u
from mhealpy import HealpixMap, HealpixBase
import time
import logging
logger = logging.getLogger(__name__)

# PyTorch and torch_geometric imports:
# These are not cosipy requirements;
# user needs to install if they want to use this class.
try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, GraphUNet as PyGGraphUNet

except: 
    logger.warning("PyTorch and PyTorch-Geometric are required for the continnuum background estimation.")
    sys.exit()

class GCN(nn.Module):
     
    def __init__(self, in_channels=1, hidden_channels=32):
       

        """
        Initialize a 4-layer Graph Convolutional Network (GCN) for node-level regression tasks.

        This model consists of three hidden graph convolutional layers with ReLU activation functions,
        followed by a final output layer without activation. Each convolutional layer aggregates
        information from neighboring nodes using the graph structure.

        Parameters
        ----------
        in_channels : int, optional
            Number of input features per node. Default is 1.
        hidden_channels : int, optional
            Number of hidden features (output channels) in each intermediate convolutional layer.
            Default is 32.

        Architecture
        ------------
        Layer 1: GCNConv(in_channels → hidden_channels) + ReLU
        Layer 2: GCNConv(hidden_channels → hidden_channels) + ReLU
        Layer 3: GCNConv(hidden_channels → hidden_channels) + ReLU
        Layer 4: GCNConv(hidden_channels → 1)  # Output layer (no activation)

        Returns
        -------
        None
        """

        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, 1)

        return

    def forward(self, x, edge_index):

        """
        Perform a forward pass through the Graph Convolutional Network (GCN).

        Applies a sequence of graph convolutional layers with ReLU activations 
        to the input node features. The final layer produces the output feature 
        representations without an activation function.

        Parameters
        ----------
        x : torch.Tensor: 
            Node feature matrix of shape (N, F_in), where 
            N is the number of nodes and F_in is the number of 
            input features per node.
        edge_index : torch.LongTensor
            Graph connectivity in COO format 
            with shape (2, E), where E is the number of edges.

        Returns
        -------
        torch.Tensor: 
            Output node feature matrix of shape (N, F_out) 
            after the final convolution layer.
        """

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))
        
        return self.conv4(x, edge_index)

class ContinuumEstimationNN(BinnedBackgroundInterface):
    
    def __init__(self):
        
        """
        COSI Background Inpainting using PyTorch Geometric (Hybrid / Self-Supervised)

        Modes:
        - supervised: uses background model as ground truth
        - self: random masking (no model)
        - hybrid: combination of both (falls back to self if no model provided)

        Includes accuracy evaluation comparing inpainted results to true background.
        """
       
        return

    def select_device(self):

        """Device selection with GPU compatibility check."""
    
        if torch.cuda.is_available():
            
            gpu_count = torch.cuda.device_count()
            logger.info(f"[INFO] GPUs available: {gpu_count}")
            capability = torch.cuda.get_device_capability(0)
            major, minor = capability
            
            # check capatability (may need to be generalized):
            if major < 3 or (major == 3 and minor < 7):
                logger.info(f"[WARNING] GPU {torch.cuda.get_device_name(0)} "
                  f"(capability {major}.{minor}) is not supported by this PyTorch build.")
                logger.info("[INFO] Falling back to CPU.")
                return torch.device('cpu')
            
            else:
                logger.info(f"[INFO] Using GPU 0: {torch.cuda.get_device_name(0)} (capability {major}.{minor})")
                if gpu_count > 1:
                    logger.info("[INFO] Multi-GPU training not enabled; using a single GPU.")
                return torch.device('cuda')
        
        else:
            logger.info("[INFO] No GPU detected. Using CPU.")
            return torch.device('cpu')

    @staticmethod
    def load_projected_data(h5_file): 

        """Loads h5 data file and projects to Em, Phi, and Psichi.
        This is the input file with the total data used to estimate 
        the background. 
        
        Parameters
        ----------
        h5_file : str
            Full path to background h5 file. 

        Returns
        -------
        projected : histogram
            Histogram object projected onto Em, Phi, and PsiChi.
        data : array
            Contents of histogram given as an array. 
        """
        
        hist = Histogram.open(h5_file)
        projected = hist.project(['Em', 'Phi', 'PsiChi'])
        data = projected.contents.todense()
        
        return projected, data

    @staticmethod
    def load_projected_psr(psr_file):
        
        """Loads precomputed point source response and projects to Em, Phi, and Psichi.
        
        Parameters
        ----------
        psr_file : str
            Full path to precomputed point source response file. 

        Returns
        -------
        projected : histogram
            Histogram object projected onto Em, Phi, and PsiChi.
        data : array
            Contents of histogram given as an array. 
        """
        
        logger.info("...loading the pre-computed point source response ...")
        psr = DetectorResponse.open(psr_file)
        logger.info("--> done")
        
        projected = psr.project(['Em', 'Phi', 'PsiChi'])
        data = projected.contents.value
        
        return projected, data

    @staticmethod
    def load_projected_model(model_file):
        
        """Loads model file and projects to Em, Phi, and Psichi.
        
        Parameters
        ----------
        model_file : str
            Full path to model h5 file. 

        Returns
        -------
        data : array
            Contents of histogram object given as an array. 
        """

        hist = Histogram.open(model_file)
        projected = hist.project(['Em', 'Phi', 'PsiChi'])
        data = projected.contents.todense()
        
        return data

    @staticmethod
    def mask_from_cumdist_vectorized(psr_map, containment=0.4):
        
        """Masks point source cone in CDS based on PSR and ARM.
        
        Parameters
        ----------
        psr_map : array
            Point source response array, i.e. contents 
            of histogram from load_projected_psr method. 
        containment : float
            Fraction of ARM to mask (default is 0.4). 
        
        Returns
        -------
        mask : array
            Boolean array defining mask. 
        """

        psr_norm = psr_map / np.sum(psr_map, axis=-1, keepdims=True)
        sort_idx = np.argsort(psr_norm, axis=-1)[..., ::-1]
        sorted_vals = np.take_along_axis(psr_norm, sort_idx, axis=-1)
        cumsum_vals = np.cumsum(sorted_vals, axis=-1)
        mask_sorted = (cumsum_vals < containment).astype(float)
        mask = np.empty_like(mask_sorted)
        np.put_along_axis(mask, sort_idx, mask_sorted, axis=-1)
        # Invert so brightest pixels = 0 (mask), background = 1 (keep)
        mask = 1 - mask
        
        return mask

    def build_healpix_graph(self, nside):
        
        """Build HEALPix graph.
        
        Parameters
        ----------
        nside : int
            nside of healpix map.

        edge_index : tensor
            Healpix graph (nodes and edges) given as tensor object. 
        """

        edges = []
        npix = hp.nside2npix(nside)
        for i in range(npix):
            neighbors = hp.get_all_neighbours(nside, i)
            for n in neighbors:
                if n >= 0:
                    edges.append([i, n])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return edge_index

    def train_inpaint(self, input_data_map, mask_map, nside,
                  mode='self', model_map=None,
                  epochs=800, lr=1e-3,
                  self_mask_fraction=0.1,
                  lambda_sup=0.5, lambda_self=0.5,
                  model_type="gcn"):

        """Training function for inpainting.
        
        Parameters
        ----------
        input_data_map : array
            Input file with total data. Returned from 
            load_projected_data method. 
        mask_map : array
            Boolean array specifying mask. Returned from
            mask_from_cumdist_vectorized method. 
        nside : int
            Nside of background map.
        mode : str, optional
            Training mode to use (default is self). Choices are:
            hybrid: uses both supervised and self.
            supervised: trains from input model map.
            self: trains from random masking. 
        model_map : array, optional
            Input model map array. Returned from 
            load_projected_model method.  
        epochs : int, optional
            Number of training epochs. Default is 800. 
        lr : float, optional
            Learning rate. Default is 1e-3. 
        self_mask_fraction : float, optional
            Fraction of pixels to randomly mask in self training mode. 
        lambda_sup : float, optional
            Weight for supervised training loss function used in 
            hybride mode. Default is 0.5. 
        lambda_self : float, optional
            Weight for self-supervised training loss function used in
            hybride mode. Default is 0.5.
        model_type : str, optional 
            Which model to use. Options are gcn (default) or unet.
        
        Returns
        -------
        inpainted : array
            Inpainted background map. 
        """
        
        # Initiate device, either CPU or GPU if available. 
        self.device = self.select_device()

        npix = hp.nside2npix(nside)
        edge_index = self.build_healpix_graph(nside).to(self.device)
    
        if model_type == "unet":
            model = PyGGraphUNet(in_channels=1, hidden_channels=32, out_channels=1, depth=3).to(self.device)
            logger.info("[INFO] Using GraphUNet model.")
        else:
            model = GCN(in_channels=1, hidden_channels=32).to(self.device)
            logger.info("[INFO] Using GCN model.")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        inpainted = np.zeros_like(input_data_map)
        n_energy, n_phi, _ = input_data_map.shape

        for e in range(n_energy):
            for p in range(n_phi):
                y = input_data_map[e, p].reshape(-1)
                mask = mask_map[e, p].reshape(-1)

                # Masked input
                x = (y * mask).reshape(-1, 1)

                # Convert to tensors
                data = Data(x=torch.tensor(x, dtype=torch.float32).to(self.device),
                        edge_index=edge_index)
                y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
                mask_tensor = torch.tensor(mask, dtype=torch.float32).to(self.device)

                target_model = None
                if model_map is not None:
                    target_model = torch.tensor(model_map[e, p], dtype=torch.float32).to(self.device).reshape(-1)

                # Training loop
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    pred = model(data.x, data.edge_index).squeeze()

                    loss_total = 0.0

                    # Supervised component
                    if mode in ['supervised', 'hybrid'] and target_model is not None:
                        sup_loss = loss_fn(pred * (1 - mask_tensor),
                                       target_model * (1 - mask_tensor))
                        loss_total += lambda_sup * sup_loss

                    # Self-supervised component
                    if mode in ['self', 'hybrid']:
                        unmasked_idx = np.where(mask == 1)[0]
                        num_rand = max(1, int(len(unmasked_idx) * self_mask_fraction))
                        rand_mask_idx = np.random.choice(unmasked_idx, num_rand, replace=False)

                        mask_self = mask.copy()
                        mask_self[rand_mask_idx] = 0
                        mask_self_tensor = torch.tensor(mask_self, dtype=torch.float32).to(self.device)

                        self_loss = loss_fn(pred * (1 - mask_self_tensor),
                                        y_tensor * (1 - mask_self_tensor))
                        loss_total += lambda_self * self_loss

                    loss_total.backward()
                    optimizer.step()

                    if epoch % 50 == 0:
                        logger.info(f"[E{e} Phi{p}] Epoch {epoch}/{epochs}: Loss {loss_total.item():.6f}")

                # Combine prediction with unmasked data
                pred_np = pred.detach().cpu().numpy().reshape(npix)
                inpainted[e, p] = y * mask + pred_np * (1 - mask)

        return inpainted

    @staticmethod
    def save_inpainted_histpy(projected_hist, inpainted_maps, output_file="inpainted.h5"):
        
        """Save results.
        
        projected_hist : histogram 
            Histrogram object. 
        inpainted_maps : array
            Inpainted background map. 
        output_file : str, optional
            Name of output histogram file. Default is inpainted.h5.
        """

        hist = Histogram(projected_hist.axes, contents=inpainted_maps, sparse=True)
        hist.write(output_file, overwrite=True)
        logger.info(f"Inpainted histogram saved to {output_file}")

        return

    def write_csv(self, x_data, y_data, save_prefix, x_label="x", y_label="y"):
        
        """Save dataframe to file. 
        
        Parameters
        ----------
        x_data : array or list
            Data for first axis.     
        y_data : array or list
            Data for seconds axis. 
        save_prefix : str
            Prefix of saved file. Will be saved as .dat file. 
        x_label : str
            Name of x axis in dat file. Default is 'x'. 
        y_label : str
            Name of y axis in dat file. Default is 'y'.
        """
        
        d = {x_label:x_data,y_label:y_data}
        df = pd.DataFrame(data=d)
        df.to_csv(f"{save_prefix}.dat",float_format='%10.5e',index=False,sep="\t",columns=[x_label,y_label])

        return

    def evaluate_inpainting_accuracy(self, true_bg, estimated_bg, prefix="eval", show_plots=False):

        """Evaluates accuracy of inpainted maps by making series of 
        comparison plots, which are saved to file and can also be shown.  

        Parameters
        ---------
        true_bg : hist
            Histogram object projected onto Em, Phi, PsiChi with the true background. 
        estimated_bg : hist
            Histogram object projected onto Em, Phi, PsiChi with the estimated background.
        prefix : str
            Prefix of saved plots. 
        show_plots : bool
            Option to show plots (default is False).
        """

        # Compare projection onto measured energy axis:
        energy = true_bg.axes['Em'].centers
        energy_err = true_bg.axes['Em'].widths / 2.0

        true_plot = true_bg.project('Em').contents.todense()
        est_plot = estimated_bg.project('Em').contents.todense()

        plt.loglog(energy,true_plot,ls="",marker="o",color="darkorange",label="BG true")
        plt.errorbar(energy,true_plot,xerr=energy_err,color="darkorange",ls="",marker="o",label="_nolabel_")

        plt.loglog(energy,est_plot,ls="",marker="s",color="cornflowerblue",label="BG estimated")
        plt.errorbar(energy,est_plot,xerr=energy_err,color="cornflowerblue",ls="",marker="s",label="_nolabel_")

        plt.xlabel("Em [keV]", fontsize=12)
        plt.ylabel("Counts", fontsize=12)
        plt.xlim(1e2,1e4)
        plt.ylim(1e5,1e9)
        plt.legend(loc=1,frameon=False)
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_em_counts.png", dpi=150)
        plt.close()

        # Plot fractional difference:
        diff = (est_plot - true_plot) / true_plot

        plt.semilogx(energy,diff,ls="",marker="o",color="black")
        plt.errorbar(energy,diff,xerr=energy_err,ls="",marker="o",color="black")

        plt.xlabel("Em [keV]", fontsize=12)
        plt.ylabel("(estmated - true)/true", fontsize=12)
        plt.xlim(1e2,1e4)
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_em_frac_diff.png", dpi=150)
        plt.close()

        # save data:
        self.write_csv(energy,diff,"Em_diff",x_label="Em[keV]",y_label="diff")

        # Compare projection onto phi axis:
        phi = true_bg.axes['Phi'].centers
        phi_err = true_bg.axes['Phi'].widths / 2.0

        true_plot = true_bg.project('Phi').contents.todense()
        est_plot = estimated_bg.project('Phi').contents.todense()

        plt.semilogy(phi,true_plot,ls="",marker="o",color="darkorange",label="BG true")
        plt.errorbar(phi,true_plot,xerr=phi_err,color="darkorange",ls="",marker="o",label="_nolabel_")

        plt.semilogy(phi,est_plot,ls="",marker="s",color="cornflowerblue",label="BG estimated")
        plt.errorbar(phi,est_plot,xerr=phi_err,color="cornflowerblue",ls="",marker="s",label="_nolabel_")

        plt.xlabel("Phi [deg]", fontsize=12)
        plt.ylabel("Counts", fontsize=12)
        plt.xlim(0,200)
        plt.ylim(1e5,1e9)
        plt.legend(loc=1,frameon=False)
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_phi_counts.png", dpi=150)
        plt.close()

        # Plot fractional difference:
        diff = (est_plot - true_plot) / true_plot

        plt.plot(phi,diff,ls="",marker="o",color="black")
        plt.errorbar(phi,diff,xerr=phi_err,ls="",marker="o",color="black")

        plt.xlabel("Phi [deg]", fontsize=12)
        plt.ylabel("(estmated - true)/true", fontsize=12)
        plt.xlim(0,200)
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_phi_frac_diff.png", dpi=150)
        plt.close()

        # save data:
        self.write_csv(phi,diff,"phi_diff",x_label="phi[deg]",y_label="diff")

        # Compare projection onto psichi:
        true_plot = true_bg.project('PsiChi')
        true_plot_map = HealpixMap(base = HealpixBase(npix = true_plot.nbins), data = true_plot.contents.todense())
        plot,ax = true_plot_map.plot('mollview')
        plt.title("True BG")
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_psichi_true_bg.pdf")
        plt.close()

        est_plot = estimated_bg.project('PsiChi')
        est_plot_map = HealpixMap(base = HealpixBase(npix = est_plot.nbins), data = est_plot.contents.todense())
        plot,ax = est_plot_map.plot('mollview')
        plt.title("Estimated BG")
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_psichi_estimated_bg.pdf")
        plt.close()

        # Calculate percent diff:
        diff = (est_plot - true_plot) / true_plot
        diff_plot_map = HealpixMap(base = HealpixBase(npix = diff.nbins), data = diff.contents.todense())
        plot,ax = diff_plot_map.plot('mollview', **{"cmap":"bwr","vmin":-0.3,"vmax":0.3})
        plt.title("Comparison")
        if show_plots == True:
            plt.show()
        plt.savefig(f"{prefix}_proj_psichi_frac_diff.pdf")
        plt.close()

        # save data:
        array = diff.contents.todense()
        self.write_csv(np.arange(array.shape[0]),array,"psichi_diff",x_label="psichi",y_label="diff")

        logger.info(f"Accuracy plots saved with prefix '{prefix}_...'")

        return

    @staticmethod
    def visualize_and_save(input_data_map, mask_map, inpainted_maps, em_bin=2, phi_bin=4, prefix="inpainted"):
        
        """
        Show and save Mollweide projections of true, masked, and inpainted maps.
        
        Parameters
        ----------
        input_data_map : array
            Input file with total data. Returned from 
            load_projected_data method. 
        mask_map : array
            Boolean array specifying mask. Returned from
            mask_from_cumdist_vectorized method.  
        inpainted_maps : array
            Estimated background. 
        em_bin : int
            Which em bin to use (default is 0).
        phi_bin : int
            Which phi bin to use (default is 0). 
        prefix : str
            Prefix of saved files.
        """
        
        maps = [
            (input_data_map[em_bin, phi_bin], f"{prefix}_true_E{em_bin}_Phi{phi_bin}.pdf", "Input Map"),
            (input_data_map[em_bin, phi_bin] * mask_map[em_bin, phi_bin], f"{prefix}_masked_E{em_bin}_Phi{phi_bin}.pdf", "Masked Map"),
            (inpainted_maps[em_bin, phi_bin], f"{prefix}_inpainted_E{em_bin}_Phi{phi_bin}.pdf", "Inpainted Map")
    ]

        for data, filename, title in maps:
            hp.mollview(data, title=title)
            plt.savefig(filename)
            plt.close()
            logger.info(f"Saved visualization: {filename}")

        return

    def estimate_bg(self, input_data, psr_file, background_model=None, 
            training_mode="self", containment=0.6, epochs=800, model_type="gcn",
            lr=1e-3, self_mask_fraction=0.1, lambda_sup=0.5, lambda_self=0.5,  
            prefix="inpainted", visualize=False, em_bin=2, phi_bin=4, 
            evaluate_only=False, true_file=None, inpainted_file=None,
            evaluate=False, show_plots=False):

        """Convenience function for estimating the background. 
        
        Parameters
        ----------
        input_data : str
            Path to HDF5 file with the input data that will be used 
            to estimate the background.
        psr_file : str
            Path to point source response HDF5 file. 
        background_model : hist, optional
            Optional background model HDF5 file. 
        training_mode : str, optional
            Training mode to use. There are three options:  
            hybrid (combination of supervised and self-supervised), 
            supervised (training based on input model), 
            and self-supervised (training based on randomly masking a fraction of the sky). 
            Default is self-supervised.
        containment : float, optional
            Containment fraction for mask. Default is 0.6. 
        epochs : int, optional 
            Number of training epochs. Default is 800.
        model_type : str, optional
            Type of NN to use. Options are gcn or unet. Default is gcn.
        lr : float, optional
            Learning rate. Default is 1e-3. 
        self_mask_fraction : float, optional
            Fraction of pixels to mask in self-supervised mode. Default is 0.1. 
        lambda_sup : float, optional
            Weight of loss function of supervised mode for hybrid 
            training. Default is 0.5.
        lambda_self : float, optional 
            Weight of loss function of self-supervised mode for hybrid 
            training. Default is 0.5.
        prefix : str, optional
            Prefix for output files. Default is 'inpainted'.
        visualize : boolean, optional
            Visualize Mollweide plots.
        em_bin : int, optional
            Energy bin index for visualization. Default is 2. 
        phi_bin : int, optional
            Phi bin index for visualization. Default is 4. 
        evaluate_only : boolean, optional
            Skip training and evaluate two histograms. Requires 
            true file and inpainted file. 
        true_file : hist, optional
            True histogram file (for evaluate-only). Default is None. 
        inpainted_file : hist, optional
            Inpainted histogram file (for evaluate-only). Default is None.
        evaluate : boolean
            Evaluate after training (inline). Default is False. 
        show_plots : boolean
            Display plots to screen. Default is False. 
        """
        
        # Record run time:
        start_time = time.time()

        # --- Evaluation-only mode ---
        if evaluate_only == True:
            if not true_file or not inpainted_file:
                raise ValueError("--evaluate-only requires --true-file and --inpainted-file")
            true_hist = Histogram.open(true_file)
            inpainted_hist = Histogram.open(inpainted_file)
            self.evaluate_inpainting_accuracy(true_hist, 
                    inpainted_hist, prefix=prefix, show_plots=show_plots)
        
            return

        # --- Training + optional inline evaluation ---
        if training_mode == "hybrid" and background_model is None:
            logger.info("[INFO] Hybrid mode selected but no background model provided.")
            logger.info("[INFO] Falling back to self-supervised loss only.")

        # Load data
        input_data_proj, input_data_map = self.load_projected_data(input_data)
        _, psr_map = self.load_projected_psr(psr_file)

        # Optional background model
        model_map = None
        if background_model:
            model_map = self.load_projected_model(background_model)

        # Derive nside
        npix = input_data_map.shape[-1]
        nside = hp.npix2nside(npix)

        # Mask from PSR
        mask_map = self.mask_from_cumdist_vectorized(psr_map, containment=containment)

        # Train & inpaint
        self.inpainted_maps = self.train_inpaint(input_data_map, mask_map, nside, 
            mode=training_mode, model_map=model_map,
            epochs=epochs, model_type=model_type,
            lr=lr, self_mask_fraction=self_mask_fraction, 
            lambda_sup=lambda_sup, lambda_self=lambda_self)

        # Save result
        self.save_inpainted_histpy(input_data_proj, self.inpainted_maps, output_file=f"{prefix}_estimated_bg.h5")

        # Visualization
        if visualize:
            self.visualize_and_save(input_data_map, mask_map, self.inpainted_maps,
                           em_bin=em_bin, phi_bin=phi_bin, prefix=prefix)

        # Inline evaluation
        if evaluate:
            inpainted_hist = Histogram(input_data_proj.axes, contents=self.inpainted_maps, sparse=True)
            self.evaluate_inpainting_accuracy(input_data_proj, inpainted_hist, prefix=prefix, show_plots=show_plots)

        end_time = time.time()
        logger.info(f"Total time elapsed: {end_time - start_time:.2f} seconds")

        return

    def load_estimated_bg(self, estimated_bg_file):
        
        """Loads inpainted histrogram from h5 file.
        
        Parameters
        ----------
        inpainted_file : str
            Path to input h5 file with estimated background.
            
        """
        
        self.inpainted_map = Histogram.open(estimated_bg_file)

        return

    def set_parameters(self, **parameters:Dict[str, u.Quantity]) -> None:
        
        self.norm = parameters['norm']

    def expectation(self, axes:Axes, copy: Optional[bool])->Histogram:
        
        return self.norm*self.inpainted_map
