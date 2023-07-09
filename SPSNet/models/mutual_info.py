import torch
import torch.nn as nn
import numpy as np

# from .neurite_pkg import soft_quantize

# def unstack(x, dim=0):
#     """
#     a useful function to simplify code paragreph in channel_wise, which trys to replace
#     tensorflow map_fn function
#     this pytorch unstack is used the same as unstack in tensorflow
#     """
#     return map(lambda x:x.squeeze(0), x.split(1, dim=dim))



class MutualInformation:
    def __init__(self,
                 bin_centers=None,
                 nb_bins=None,
                 soft_bin_alpha=None,
                 min_clip=None,
                 max_clip=None):
        """
        Initialize the mutual information class

        Arguments below are related to soft quantizing of volumes, which is done automatically 
        in functions that comptue MI over volumes (e.g. volumes(), volume_seg(), channelwise()) 
        using these parameters

        Args:
            bin_centers (np.float32, optional): Array or list of bin centers. Defaults to None.
            nb_bins (int, optional):  Number of bins. Defaults to 16 if bin_centers
                is not specified.
            soft_bin_alpha (int, optional): Alpha in RBF of soft quantization. Defaults
                to `1 / 2 * square(sigma)`.
            min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
            max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
        """

        self.bin_centers=None
        if bin_centers is not None:
            self.bin_centers = torch.tensor(bin_centers).float()
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]

        self.nb_bins = nb_bins
        if bin_centers is None and nb_bins is None:
            self.nb_bins = 16

        self.min_clip = min_clip
        if self.min_clip is None:
            self.min_clip = -np.inf

        self.max_clip = max_clip
        if self.max_clip is None:
            self.max_clip = np.inf

        self.soft_bin_alpha = soft_bin_alpha
        if self.soft_bin_alpha is None:
            sigma_ratio = 0.5
            if self.bin_centers is None:
                sigma = torch.tensor(sigma_ratio / (self.nb_bins - 1)).float()
            else:
                sigma = sigma_ratio * torch.mean(torch.diff(bin_centers))
            self.soft_bin_alpha = 1 / (2 * torch.square(sigma))
            print(self.soft_bin_alpha)
    
    def volumes(self, x, y, weight=None):
        """
        Mutual information for each item in a batch of volumes. 

        Algorithm: 
        - use soft_quantize() to create a soft quantization (binning) of 
          intensities in each channel
        - channelwise()

        Parameters:
            x and y:  [bs, ..., N_channels]
            weight: [N_channels]

        Returns:
            Tensor of size [bs]
        """
        # check shapes
        assert x.shape[-1] == y.shape[-1], \
            f'volume_mi requires two single-channel volumes. See channelwise().\
                the x shape is {x.shape} and y shape is {y.shape}'
        if weight is not None:
            weight = np.array(weight)
            assert weight.shape[0] == x.shape[-1], \
                f'weight shape is {weight.shape} but x shape is {x.shape}'

        mean_mi = 0
        if weight is not None:
            for i, (x_single, y_single) in enumerate(zip(x.split(1, dim=-1), y.split(1, dim=-1))):
                mean_mi += weight[i]*torch.flatten(self.channelwise(x_single,y_single)).mean()/x.shape[-1]/weight.sum()
                # just use the sum item as the weight.sum() is a numpy array
        else:
            for x_single, y_single in zip(x.split(1, dim=-1), y.split(1, dim=-1)):
                mean_mi += torch.flatten(self.channelwise(x_single,y_single)).mean()/x.shape[-1]
        # volume mi
        return mean_mi
    

    def channelwise(self, x, y):
        """
        Mutual information for each channel in x and y. Thus for each item and channel this 
        returns retuns MI(x[...,i], x[...,i]). To do this, we use soft_quantize() to 
        create a soft quantization (binning) of the intensities in each channel

        Parameters:
            x and y:  [bs, ..., C]

        Returns:
            Tensor of size [bs, C]
        """
        # check shapes
        assert x.shape == y.shape, 'volume shapes do not match'

        # reshape to [bs, V, C]
        if x.dim() != 3:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            y = y.reshape(y.shape[0], -1, y.shape[-1])

        # move channels to first dimension
        cx = x.permute(2,0,1)
        cy = y.permute(2,0,1)                               # [C, bs, V]

        # soft quantize
        cxq = self._soft_sim_map(cx)                        # [C, bs, V, B]
        cyq = self._soft_sim_map(cy)                        # [C, bs, V, B]

        # get mi
        cout = self.maps(cxq.squeeze(0), cyq.squeeze(0)).unsqueeze(0)  # [C, bs] # only C == 1 is OK
        # print(cout.shape)
        # permute back
        return cout.permute(1, 0) 


    def _soft_sim_map(self, x):
        """
        soft quantization of intensities (values) in a given volume

        See neurite.utils.soft_quantize

        Parameters:
            x [C, bs, ...]: intensity image. 

        Returns:
            volume with one more dimension [C, bs, ..., B]
        """

        return self.soft_quantize(x,
                                alpha=self.soft_bin_alpha,
                                bin_centers=self.bin_centers,
                                nb_bins=self.nb_bins,
                                min_clip=self.min_clip,
                                max_clip=self.max_clip,
                                return_log=False)
    
    def soft_quantize(self,
                    x,
                    bin_centers=None,
                    nb_bins=16,
                    alpha=1,
                    min_clip=-np.inf,
                    max_clip=np.inf,
                    return_log=False):
        """
        (Softly) quantize intensities (values) in a given volume, based on RBFs. 
        In numpy this (hard quantization) is called "digitize".

        Specify bin_centers OR number of bins 
            (which will estimate bin centers based on a heuristic using the min/max of the image)

        Algorithm: 
        - create (or obtain) a set of bins
        - for each array element, that value v gets assigned to all bins with 
            a weight of exp(-alpha * (v - c)), where c is the bin center
        - return volume x nb_bins

        Parameters:
            x [C, bs, ...]: intensity image. 
            bin_centers (torch.tensor): bin centers for soft histogram.
                Defaults to None.
            nb_bins (int, optional): number of bins, if bin_centers is not specified. 
                Defaults to 16.
            alpha (int, optional): alpha in RBF.
                Defaults to 1.
            min_clip (float, optional): Lower value to clip data. Defaults to -np.inf.
            max_clip (float, optional): Upper value to clip data. Defaults to np.inf.
            return_log (bool, optional): [description]. Defaults to False.

        Returns:
            torch.tensor (float32): volume with one more dimension [bs, ..., B]

        If you find this function useful, please consider citing:
            M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
            SynthMorph: learning contrast-invariant registration without acquired images
            IEEE Transactions on Medical Imaging (TMI), in press, 2021
            https://doi.org/10.1109/TMI.2021.3116879
        """

        if bin_centers is not None:
            bin_centers = bin_centers.float()
            assert nb_bins is None, 'cannot provide both bin_centers and nb_bins'
            nb_bins = bin_centers.shape[0]
        else:
            if nb_bins is None:
                nb_bins = 16
            # get bin centers dynamically
            # TODO: perhaps consider an option to quantize by percentiles:
            #   minval = tfp.stats.percentile(x, 1)
            #   maxval = tfp.stats.percentile(x, 99)
            bin_centers = torch.linspace(x.min().item(), x.max().item(), nb_bins)

        # clipping at bin values
        x = x.unsqueeze(-1)       # (C, bs, V, 1)                                       # [..., 1]
        x = torch.clamp(x, min_clip, max_clip)

        # reshape bin centers to be (1, 1, 1, B)
        new_shape = [1] * (x.dim() - 1) + [nb_bins]
        bin_centers = bin_centers.reshape(new_shape).to(x.device)                     # [1, 1, ..., B]

        # compute image terms
        # TODO: need to go to log space? not sure
        bin_diff = torch.square(x - bin_centers)        #broadcast, (C, bs, V, B)
        log = -alpha * bin_diff                                               # (C, bs, V, B)

        if return_log:
            return log                                               # (C, bs, V, B)
        else:
            return torch.exp(log)                                                 # (C, bs, V, B)

    def maps(self, x, y, eps=1e-7):
        """
        Computes mutual information for each entry in batch, assuming each item contains 
        probability or similarity maps *at each voxel*. These could be e.g. from a softmax output 
        (e.g. when performing segmentaiton) or from soft_quantization of intensity image.

        Note: the MI is computed separate for each itemin the batch, so the joint probabilities 
        might be  different across inputs. In some cases, computing MI actoss the whole batch 
        might be desireable (TODO).

        Parameters:
            x and y are probability maps of size [bs, ..., B], where B is the size of the 
              discrete probability domain grid (e.g. bins/labels). B can be different for x and y.
            each channel of the image pair has a pair of (x,y) and are put into this function together.
        Returns:
            Tensor of size [bs]
        """

        # check shapes
        assert x.shape == y.shape, 'two image shape should be the same in self.map funtion'
        # tf.debugging.assert_non_negative(x)
        # tf.debugging.assert_non_negative(y)
        # TODO the intensity image is not all positive, check whether this matters

        # reshape to [bs, V, B]
        if x.dim() != 3:
            x = x.reshape(x.shape[0], -1, x.shape[-1])               # [bs, V, B1]      
            y = y.reshape(y.shape[0], -1, y.shape[-1])               # [bs, V, B2]

        # joint probability for each batch entry
        x_trans = x.permute(0, 2, 1)                                 # [bs, B1, V]
        pxy = torch.bmm(x_trans, y)                                  # [bs, B1, B2]
        pxy = pxy / (pxy.sum(dim=[1, 2], keepdim=True) + eps)        # [bs, B1, B2]

        # x probability for each batch entry
        px = x.sum(dim=1, keepdim=True)                              # [bs, 1, B1]
        px = px / (px.sum(dim=2, keepdim=True) + eps)                # [bs, 1, B1]

        # y probability for each batch entry
        py = y.sum(dim=1, keepdim=True)                              # [bs, 1, B2]
        py = py / (py.sum(dim=2, keepdim=True) + eps)                # [bs, 1, B2]

        # independent xy probability
        px_trans = px.permute(0, 2, 1)                               # [bs, B1, 1]
        pxpy = torch.bmm(px_trans, py)                               # [bs, B1, B2]
        pxpy_eps = pxpy + eps

        # mutual information
        log_term = torch.log(pxy / pxpy_eps + eps)                   # [bs, B1, B2]
        mi = torch.sum(pxy * log_term, dim=[1, 2])                   # [bs]
        return mi