import torch
import numpy as np

TORCH_VERSION = torch.__version__

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]]).astype("float32")

max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))

color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

color_mean = [0.48, 0.46, 0.41]

class Image():
    def __init__(
            self, 
            w,
            h=None, 
            sd=None, 
            batch=None, 
            decorrelate=False,
            fft:bool=True, 
            channels:int=None,
            decay_power=1,
            device=None,
        ):
        h = h if h is not None else w
        batch = batch if batch is not None else 1
        ch = channels if channels is not None else 3
        self.channels = channels
        self.shape = [batch, ch, h, w]

        self.decorrelate = decorrelate
        self.sd = sd if sd is not None else 0.01
        self.fft = fft
        self.decay_power = decay_power
        self.device = 'cpu' if device is None else device

        self.scale = None
        if(self.fft):
            self._param_tensor, self.scale = self._create_fft_image()
            self._image_f = self._ftt_image()
        else:
            self._param_tensor = self._create_pixel_image()
            self._image_f = self._pixel_image()

    def to(self, device):
        self.device = device
        self._param_tensor = self._param_tensor.detach().to(device).requires_grad_(True)
        if(self.scale is not None):
            self.scale = self.scale.to(device)
        return self

    def _create_fft_image(self) -> torch.Tensor | torch.Tensor:
        """
            Return param tensor and scale
        """
        batch, channels, h, w = self.shape
        freqs = Image.rfft2d_freqs(h, w)
        init_val_size = (batch, channels) + freqs.shape + (2,) # 2 for imaginary and real components

        spectrum_real_imag_t = (torch.randn(*init_val_size) * self.sd).to(self.device).requires_grad_(True)

        scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** self.decay_power
        scale = torch.tensor(scale).float()[None, None, ..., None].to(self.device)
        return spectrum_real_imag_t, scale

    def _create_pixel_image(self):
        tensor = (torch.randn(*self.shape) * self.sd).to(self.device).requires_grad_(True)
        return tensor

    def _pixel_image(self):
        def inner():
            return self._param_tensor
        return inner

    def _ftt_image(self):
        """
            Call this each loop to recalculate image.
        """
        def inner():
            batch, channels, h, w = self.shape
            scaled_spectrum_t = self.scale * self._param_tensor
            if TORCH_VERSION >= "1.7.0":
                import torch.fft
                if type(scaled_spectrum_t) is not torch.complex64:
                    scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
                image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
            else:
                import torch
                image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
            image = image[:batch, :channels, :h, :w]
            magic = 4.0 # Magic constant from Lucid library; increasing this seems to reduce saturation
            image = image / magic
            return image
        return inner

    def _linear_decorrelate_color(self, tensor):
        t_permute = tensor.permute(0, 2, 3, 1)
        t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(self.device))
        tensor = t_permute.permute(0, 3, 1, 2)
        return tensor

    def _to_valid_rgb(self, image):
        if self.decorrelate:
            image = self._linear_decorrelate_color(image)
        return torch.sigmoid(image)

    # From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
    def rfft2d_freqs(h, w):
        """Computes 2D spectrum frequencies."""
        fy = np.fft.fftfreq(h)[:, None]
        # when we have an odd input dimension we need to keep one additional
        # frequency and later cut off 1 pixel
        if w % 2 == 1:
            fx = np.fft.fftfreq(w)[: w // 2 + 2]
        else:
            fx = np.fft.fftfreq(w)[: w // 2 + 1]
        return np.sqrt(fx * fx + fy * fy)

    def param(self):
        return [self._param_tensor]

    def image(self):
        return self._to_valid_rgb(self._image_f())

    def param_image(self):
        return self.params(), self.image()


'''
def image(w, h=None, sd=None, batch=None, decorrelate=True,
          fft=True, channels=None):
    h = h if h is not None else w
    batch = batch if batch is not None else 1
    ch = channels if channels is not None else 3
    shape = [batch, ch, h, w]
    param_f = fft_image if fft else pixel_image
    params, image_f = param_f(shape, sd=sd)
    if channels:
        output = to_valid_rgb(image_f, decorrelate=False)
    else:
        output = to_valid_rgb(image_f, decorrelate=decorrelate)
    return params, output

def _linear_decorrelate_color(tensor):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_permute = tensor.permute(0, 2, 3, 1)
    t_permute = torch.matmul(t_permute, torch.tensor(color_correlation_normalized.T).to(device))
    tensor = t_permute.permute(0, 3, 1, 2)
    return tensor

def to_valid_rgb(image_f, decorrelate=False):
    def inner():
        image = image_f()
        if decorrelate:
            image = _linear_decorrelate_color(image)
        return torch.sigmoid(image)
    return inner


def pixel_image(shape, sd=None):
    sd = sd or 0.01
    tensor = (torch.randn(*shape) * sd).to(device).requires_grad_(True)
    return [tensor], lambda: tensor


# From https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/spatial.py
def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""
    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)


def fft_image(shape, sd=None, decay_power=1):
    batch, channels, h, w = shape
    freqs = rfft2d_freqs(h, w)
    init_val_size = (batch, channels) + freqs.shape + (2,) # 2 for imaginary and real components
    sd = sd or 0.01

    spectrum_real_imag_t = (torch.randn(*init_val_size) * sd).to(device).requires_grad_(True)

    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale = torch.tensor(scale).float()[None, None, ..., None].to(device)

    def inner():
        scaled_spectrum_t = scale * spectrum_real_imag_t
        if TORCH_VERSION >= "1.7.0":
            import torch.fft
            if type(scaled_spectrum_t) is not torch.complex64:
                scaled_spectrum_t = torch.view_as_complex(scaled_spectrum_t)
            image = torch.fft.irfftn(scaled_spectrum_t, s=(h, w), norm='ortho')
        else:
            import torch
            image = torch.irfft(scaled_spectrum_t, 2, normalized=True, signal_sizes=(h, w))
        image = image[:batch, :channels, :h, :w]
        magic = 4.0 # Magic constant from Lucid library; increasing this seems to reduce saturation
        image = image / magic
        return image
    return [spectrum_real_imag_t], inner

'''