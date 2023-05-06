import torch
import numpy as np
import torch.fft
from abc import abstractmethod
from collections.abc import Sequence

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = [0.48, 0.46, 0.41]

class _ImageFunctional():
    def __init__(self, device=None) -> None:
        self._param_tensor = None
        self._image_f = None

        self.device = 'cpu' if device is None else device

    @property
    def param_tensor(self):
        if(self._param_tensor is None):
            raise Exception('Parameter tensor was not initialized. Call reinit().')
        return self._param_tensor
    @param_tensor.setter
    def param_tensor(self, val):
        self._param_tensor = val

    @abstractmethod
    def reinit(self):
        pass

    def to(self, device):
        self.device = device
        if(self._param_tensor is not None):
            if(isinstance(self._param_tensor, Sequence)):
                self._param_tensor = [i.detach().to(device).requires_grad_(True) for i in self._param_tensor]
            else:
                self._param_tensor = self._param_tensor.detach().to(device).requires_grad_(True)
        return self

    @abstractmethod
    def param(self):
        pass

    @abstractmethod
    def image(self):
        pass

class _Image(_ImageFunctional):
    def __init__(
            self, 
            w,
            h=None, 
            sd=None, 
            batch=None, 
            decorrelate=False,
            channels:int=None,
            device=None,
        ):
        super().__init__(device=device)

        h = h if h is not None else w
        batch = batch if batch is not None else 1
        ch = channels if channels is not None else 3
        self.channels = channels
        self.shape = [batch, ch, h, w]

        self.decorrelate = decorrelate
        self.sd = sd if sd is not None else 0.01

        self._color = self._to_valid_rgb

    def _linear_decorrelate_color(self, tensor):
        t_permute = tensor.permute(0, 2, 3, 1)
        t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(self.device))
        tensor = t_permute.permute(0, 3, 1, 2)
        return tensor

    def _to_valid_rgb(self, image):
        if self.decorrelate:
            image = self._linear_decorrelate_color(image)
        return torch.sigmoid(image)
    
    def param(self):
        return [self.param_tensor]

    def image(self):
        return self._color(self._image_f())

    def param_image(self):
        return self.params(), self.image()

class FFTImage(_Image):
    def __init__(self, *args, decay_power=1, magic=4, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decay_power = decay_power
        self.magic = magic # Magic constant from Lucid library; increasing this seems to reduce saturation

        #self._create_fft_image()
        self._image_f = self._ftt_image_f

    def reinit(self):
        self._create_fft_image()

    def _create_fft_image(self) -> torch.Tensor | torch.Tensor:
        """
            Return param tensor and scale
        """
        batch, channels, h, w = self.shape
        freqs = FFTImage.rfft2d_freqs(h, w)
        init_val_size = (batch, channels) + freqs.shape + (2,) # 2 for imaginary and real components

        self.param_tensor = (torch.randn(*init_val_size) * self.sd).to(self.device).requires_grad_(True)

        scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** self.decay_power
        self.scale = torch.tensor(scale).float()[None, None, ..., None].to(self.device)

    def _ftt_image_f(self):
        """
            Call this each loop to recalculate image.
        """
        batch, channels, h, w = self.shape
        scaled_spectrum_t = self.scale * self.param_tensor
        if type(scaled_spectrum_t) is not torch.complex64:
            scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
        image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        image = image[:batch, :channels, :h, :w]
        image = image / self.magic
        return image

    def to(self, device):
        if(hasattr(self, 'scale') and self.scale is not None):
            self.scale = self.scale.to(device)
        return super().to(device)

    def rfft2d_freqs(h, w):
        # From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py

        """Computes 2D spectrum frequencies."""
        
        fy = np.fft.fftfreq(h)[:, None]
        # when we have an odd input dimension we need to keep one additional
        # frequency and later cut off 1 pixel
        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]
        return np.sqrt(fx * fx + fy * fy)

class PixelImage(_Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #self._create_pixel_image()
        self._image_f = self._pixel_image_f
    
    def reinit(self):
        self._create_pixel_image()

    def _create_pixel_image(self):
        tensor = (torch.randn(*self.shape) * self.sd).to(self.device).requires_grad_(True)
        self.param_tensor = tensor

    def _pixel_image_f(self):
        return self.param_tensor

class CustomImage(_ImageFunctional):
    def __init__(self, image_f, param_tensor, reinit_f, device=None):
        super().__init__(device=device)
        self.image_f = image_f
        self.param_tensor = param_tensor if isinstance(param_tensor, Sequence) else [param_tensor]
        self.reinit_f = reinit_f

    def param(self):
        return self.param_tensor

    def image(self):
        return self.image_f()

    def reinit(self):
        self.reinit_f(self.image_f, self.param_tensor)

class Image():
    def __new__(
        self, 
        dtype:str,
        w=None,
        h=None, 
        sd=None, 
        batch=None, 
        decorrelate=False,
        channels:int=None,
        decay_power=1,
        device=None,
        image_f=None,
        param_tensor=None,
        reinit_f=None,
    ) -> None:
        """
            dtype - fft; pixel;
        """
        print(f'VIS: Selected dream image type: {dtype}')
        if(dtype == 'fft'):
            assert w is not None, "Bad value, 'w' cannot be None for 'fft'"
            return FFTImage(w=w, h=h, sd=sd, batch=batch, decorrelate=decorrelate, channels=channels, decay_power=decay_power, device=device)
        elif(dtype == 'pixel'):
            assert w is not None, "Bad value, 'w' cannot be None for 'pixel'"
            return PixelImage(w=w, h=h, sd=sd, batch=batch, decorrelate=decorrelate, channels=channels, device=device)
        elif(dtype == 'custom'):
            assert image_f != None and param_tensor != None and reinit_f != None, "Bad value, image_f and param_tensor cannot be None for dtype 'custom'"
            return CustomImage(image_f=image_f, param_tensor=param_tensor, device=device, reinit_f=reinit_f)
        else:
            raise Exception(f'Unknown type "{dtype}"')
