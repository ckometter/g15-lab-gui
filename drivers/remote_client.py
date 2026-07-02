import Pyro4
import numpy as np
from qcodes.instrument import Instrument
from qcodes.parameters import ArrayParameter

def remote_get(proxy, property_name, method_name=None):
    """
    Get a property from a Pyro proxy.
    If a direct property/attribute exists and is not callable, return it.
    Otherwise, fall back to calling the getter method.
    """
    try:
        val = getattr(proxy, property_name)
        if not callable(val):
            return val
    except AttributeError:
        pass

    m_name = method_name or f"get_{property_name}"
    method = getattr(proxy, m_name)
    return method()

def remote_set(proxy, property_name, val, method_name=None):
    """
    Set a property on a Pyro proxy.
    If a setter method exists, call it. Otherwise, set the attribute directly.
    """
    m_name = method_name or f"set_{property_name}"
    try:
        method = getattr(proxy, m_name)
        method(val)
        return
    except AttributeError:
        pass

    setattr(proxy, property_name, val)

class PyroProxyWrapper:
    def __init__(self, proxy):
        object.__setattr__(self, '_proxy', proxy)
        object.__setattr__(self, '_cached_wavelengths', None)

    def __getattr__(self, name):
        if name == 'get_status':
            return self.get_status
        elif name == 'start_preview':
            return self.start_preview
        elif name == 'stop_preview':
            return self.stop_preview
        elif name == 'get_latest_preview_data':
            return self.get_latest_preview_data
        elif name == 'get_wavelengths_cache':
            return self.get_wavelengths_cache
        elif name == 'abort_acquisition':
            return self.abort_acquisition
        elif name == 'get_spectrum':
            return self.get_spectrum
        elif name == 'set_grating':
            return self.set_grating
        elif name == 'set_shutter_control':
            return self.set_shutter_control
        elif name == 'set_target_temperature':
            return self.set_target_temperature
        
        # Forward everything else to the real proxy
        return getattr(object.__getattribute__(self, '_proxy'), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, '_proxy'), name, value)

    def __dir__(self):
        real_proxy = object.__getattribute__(self, '_proxy')
        try:
            attrs = dir(real_proxy)
        except Exception:
            attrs = []
        return attrs + [
            'get_status', 'start_preview', 'stop_preview',
            'get_latest_preview_data', 'get_wavelengths_cache',
            'abort_acquisition', 'get_spectrum', 'set_grating',
            'set_shutter_control', 'set_target_temperature'
        ]

    def __enter__(self):
        object.__getattribute__(self, '_proxy').__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return object.__getattribute__(self, '_proxy').__exit__(exc_type, exc_val, exc_tb)

    def set_grating(self, val):
        proxy = object.__getattribute__(self, '_proxy')
        if isinstance(val, str):
            if '][0][' in val:
                g_val = 1
            elif '][1][' in val:
                g_val = 1
            elif '][2][' in val:
                g_val = 2
            else:
                try:
                    g_val = int(val)
                except ValueError:
                    g_val = 1
        else:
            g_val = int(val)
        proxy.grating = g_val

    def set_shutter_control(self, val):
        proxy = object.__getattribute__(self, '_proxy')
        proxy.shutter_control = int(val)

    def set_target_temperature(self, val):
        proxy = object.__getattribute__(self, '_proxy')
        proxy.target_temperature = float(val)

    def get_status(self):
        proxy = object.__getattribute__(self, '_proxy')
        try:
            # Safely fetch properties with default fallbacks
            temp = getattr(proxy, 'temperature', None)
            target_temp = getattr(proxy, 'target_temperature', None)
            grating_num = getattr(proxy, 'grating', None)
            exposure = getattr(proxy, 'exposure_time', None)
            wl = getattr(proxy, 'wavelength', None)
            shutter = getattr(proxy, 'shutter_control', None)
            num_frames = getattr(proxy, 'num_frames', None)

            # Map WinSpec grating number back to LightField-style string representation
            grating = None
            if grating_num == 1:
                grating = "[750  nm,1200][1][0]"
            elif grating_num == 2:
                grating = "[1.0  \u00e6m,300][2][0]"
            elif grating_num == 3:
                grating = "[hol,1500][0][0]"
            else:
                grating = grating_num

            # Map to expected structure
            data = {
                'success': True,
                'temp': temp,
                'target_temp': target_temp,
                'status': 1 if (temp is not None and target_temp is not None and abs(temp - target_temp) < 1.0) else 2,
                'gain': 1.0,
                'grating': grating,
                'exposure': exposure,
                'wl': wl,
                'shutter': shutter,
                'num_frames': num_frames
            }
            return data
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def start_preview(self):
        proxy = object.__getattribute__(self, '_proxy')
        if hasattr(proxy, "start_focus"):
            try:
                proxy.start_focus()
                return
            except Exception as e:
                print(f"Wrapper: start_focus failed, falling back: {e}")
        pass

    def stop_preview(self):
        proxy = object.__getattribute__(self, '_proxy')
        if hasattr(proxy, "stop_focus"):
            try:
                proxy.stop_focus()
                return
            except Exception as e:
                print(f"Wrapper: stop_focus failed, falling back: {e}")
        pass

    def get_latest_preview_data(self):
        proxy = object.__getattribute__(self, '_proxy')
        if hasattr(proxy, "get_focus_data"):
            try:
                data = proxy.get_focus_data()
                if data is not None:
                    return data
            except Exception as e:
                print(f"Wrapper: get_focus_data failed, falling back: {e}")
                
        try:
            # We call get_spectrum(wlen=False)
            spectra = proxy.get_spectrum(wlen=False)
            spectra_arr = np.array(spectra)
            if spectra_arr.ndim == 3:
                # Sum over ROI axis (axis 0) to get (frames, pixels), then take first frame
                spectra_arr = np.sum(spectra_arr, axis=0)
            if spectra_arr.ndim == 2:
                return spectra_arr[0].tolist()
            return spectra_arr.tolist()
        except Exception as e:
            print(f"Wrapper: failed to get latest preview data: {e}")
            return None

    def get_wavelengths_cache(self):
        # Return local cache if available
        cached = object.__getattribute__(self, '_cached_wavelengths')
        if cached is not None:
            return cached
            
        proxy = object.__getattribute__(self, '_proxy')
        try:
            # Try to get polynomial calibration first as it is much faster (<1s vs ~20s)
            if hasattr(proxy, 'get_calibration_poly'):
                wlens = proxy.get_calibration_poly()
            else:
                wlens = proxy.get_calibration()
            if wlens is not None:
                object.__setattr__(self, '_cached_wavelengths', wlens)
            return wlens
        except Exception:
            return None

    def abort_acquisition(self):
        proxy = object.__getattribute__(self, '_proxy')
        # Force-close the socket at the OS level first to instantly break the blocked read
        try:
            conn = getattr(proxy, "_pyroConnection", None)
            if conn:
                sock = getattr(conn, "sock", None)
                if sock:
                    import socket
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    try:
                        sock.close()
                    except Exception:
                        pass
        except Exception:
            pass
        # Finally release the proxy connection in Pyro
        try:
            proxy._pyroRelease()
        except Exception:
            pass

    def get_spectrum(self, *args, **kwargs):
        # Force wlenpoly=True unless explicitly overridden
        if 'wlen' not in kwargs and 'wlenpoly' not in kwargs and not (len(args) > 0):
            kwargs['wlenpoly'] = True
            
        proxy = object.__getattribute__(self, '_proxy')
        res = proxy.get_spectrum(*args, **kwargs)
        
        # If it returned a tuple of (spectra, wlens)
        if isinstance(res, tuple) and len(res) == 2:
            spectra, wlens = res
            spectra_arr = np.array(spectra)
            if spectra_arr.ndim == 3:
                # Sum over ROI axis (axis 0) to get (frames, pixels)
                spectra_arr = np.sum(spectra_arr, axis=0)
                
            # Cache the wavelengths for subsequent get_wavelengths_cache calls
            if wlens is not None:
                object.__setattr__(self, '_cached_wavelengths', wlens)
                
            return spectra_arr.tolist(), wlens
        # If it returned just spectra (wlen=False)
        spectra_arr = np.array(res)
        if spectra_arr.ndim == 3:
            # Sum over ROI axis (axis 0) to get (frames, pixels)
            spectra_arr = np.sum(spectra_arr, axis=0)
        return spectra_arr.tolist()

class NIRvanaSocketWrapper:
    def __init__(self, server_ip, port):
        self.server_ip = server_ip
        self.port = port
        self._exposure_time = 0.1  # seconds
        self._num_frames = 1
        self._cached_wavelengths = None
        self._active_socket = None

    def get_exposure_time(self):
        return self._exposure_time

    def set_exposure_time(self, val):
        self._exposure_time = float(val)

    def get_num_frames(self):
        return self._num_frames

    def set_num_frames(self, val):
        self._num_frames = int(val)

    def get_readout_mode(self):
        return "Full Frame"

    def set_readout_mode(self, val):
        pass

    def get_acquisition_mode(self):
        return "Single"

    def set_acquisition_mode(self, val):
        pass

    def get_status(self):
        grating = 'NIRvana Fixed'
        wlens = self.get_wavelengths_cache()
        if wlens is not None and len(wlens) > 1:
            avg_step = float(np.mean(np.diff(wlens)))
            if avg_step > 0.35:
                grating = "[150][0][0]"  # 150 g/mm
            elif avg_step > 0.18:
                grating = "[1.0  \u00e6m,300][2][0]"  # 300 g/mm
            elif avg_step > 0.08:
                grating = "[600][0][0]"  # 600 g/mm
            elif avg_step > 0.04:
                grating = "[750  nm,1200][1][0]"  # 1200 g/mm
            else:
                grating = "[hol,1500][0][0]"  # 1500 g/mm

        return {
            'success': True,
            'temp': -80.0,
            'target_temp': -80.0,
            'status': 1,
            'gain': 1.0,
            'grating': grating,
            'exposure': self._exposure_time,
            'wl': 1100.0,
            'shutter': 'N/A',
            'num_frames': self._num_frames
        }

    def start_preview(self):
        pass

    def stop_preview(self):
        pass

    def get_latest_preview_data(self):
        res = self.get_spectrum()
        if res is not None:
            intensities, _ = res
            intensities_arr = np.array(intensities)
            if intensities_arr.ndim == 2:
                return intensities_arr[0].tolist()
            return intensities_arr.tolist()
        return None

    def get_wavelengths_cache(self):
        if self._cached_wavelengths is not None:
            return self._cached_wavelengths
        return np.linspace(1049.17, 1130.78, 640).tolist()

    def abort_acquisition(self):
        sock = self._active_socket
        if sock:
            import socket
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

    def get_spectrum(self, *args, **kwargs):
        import socket
        import json
        from io import StringIO
        import pandas as pd
        
        spec_params = {
            "command": "start_acquisition",
            "exposure_time": self._exposure_time * 1e3,  # convert to ms
            "num_frames": self._num_frames
        }
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._active_socket = sock
        try:
            sock.settimeout(10*60)
            sock.connect((self.server_ip, self.port))
            sock.sendall(json.dumps(spec_params).encode('utf-8'))
            
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                
            data_str = b''.join(chunks).decode('utf-8')
            if not data_str.strip():
                return None
                
            df = pd.read_csv(StringIO(data_str), engine='python')
            df['Frame'] = df['Frame'].astype(int) - 1
            
            def parse_wavelength(x):
                try: return float(x)
                except: return np.nan
            df['Wavelength'] = df['Wavelength'].apply(parse_wavelength)
            df['Intensity'] = df['Intensity'].astype(np.int32)
            
            num_frames = df['Frame'].max() + 1
            num_wavelengths = df['Wavelength'].nunique()
            df = df.sort_values(['Frame', 'Wavelength'])
            
            intensity_array = df['Intensity'].to_numpy().reshape((num_frames, num_wavelengths))
            wavelength_array = df[df['Frame'] == 0]['Wavelength'].to_numpy()
            
            self._cached_wavelengths = wavelength_array.tolist()
            return intensity_array.tolist(), wavelength_array.tolist()
            
        except Exception as e:
            print(f"NIRvanaSocketWrapper error: {e}")
            return None
        finally:
            try:
                sock.close()
            except Exception:
                pass
            self._active_socket = None

class NIRvanaMonoWrapper:
    def __init__(self):
        self._wavelength = 1100.0

    def get_wavelength(self):
        return self._wavelength

    def set_wavelength(self, val):
        self._wavelength = float(val)

class RemoteCCDImage(ArrayParameter):
    def __init__(self, name, instrument, shape=(512, 512), **kwargs):
        setpoints = (np.arange(shape[0]), np.arange(shape[1]))
        super().__init__(name=name, shape=shape, instrument=instrument, unit='counts', setpoints=setpoints, **kwargs)
        
    def get_raw(self) -> np.ndarray:
        return self.instrument.capture_image()

class ClientLightFieldCamera(Instrument):
    """
    QCoDeS client driver for the Princeton Instruments Camera 
    exposed by the unified LightField Pyro4 server.
    """
    def __init__(self, name, server_ip="G15-Pylon.dhcp-int.phys.ethz.ch", port=50920, pyro_uri=None, spectrometer_type="lightfieldG15", **kwargs):
        super().__init__(name, **kwargs)
        self.spectrometer_type = spectrometer_type
        
        if spectrometer_type == 'lightfieldG11':
            self.remote_lf = NIRvanaSocketWrapper(server_ip, port)
        else:
            Pyro4.config.SERIALIZER = 'pickle'
            Pyro4.config.PICKLE_PROTOCOL_VERSION = 2
            if pyro_uri is None:
                pyro_uri = f"PYRO:LightField@{server_ip}:{port}"
            proxy = Pyro4.Proxy(pyro_uri)
            if spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
                self.remote_lf = PyroProxyWrapper(proxy)
            else:
                self.remote_lf = proxy
        
        self.add_parameter('exposure_time',
                           label='Exposure Time',
                           unit='s',
                           get_cmd=lambda: remote_get(self.remote_lf, 'exposure_time', 'get_exposure_time'),
                           set_cmd=lambda val: remote_set(self.remote_lf, 'exposure_time', val, 'set_exposure_time'))

        self.add_parameter('num_frames',
                           label='Number of Frames',
                           get_cmd=lambda: remote_get(self.remote_lf, 'num_frames', 'get_num_frames'),
                           set_cmd=lambda val: remote_set(self.remote_lf, 'num_frames', val, 'set_num_frames'))

        self.add_parameter('readout_mode',
                           label='Readout Mode',
                           get_cmd=lambda: "Full Frame",
                           set_cmd=lambda val: remote_set(self.remote_lf, 'readout_mode', val, 'set_readout_mode'))
                           
        self.add_parameter('image',
                           parameter_class=RemoteCCDImage,
                           shape=(512, 512))

    def capture_image(self) -> np.ndarray:
        # Fetch the spectrum (Note: get_spectrum returns (intensities, wavelengths))
        if self.spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
            res = self.remote_lf.get_spectrum(wlen=True)
        else:
            res = self.remote_lf.get_spectrum()
        intensities, wavelengths = res
        
        intensities_arr = np.array(intensities)
        if intensities_arr.ndim == 3:
            # WinSpec returns 3D array (ROI, frames, pixels)
            intensities_arr = np.mean(intensities_arr[0], axis=0)
        elif intensities_arr.ndim == 2:
            intensities_arr = np.mean(intensities_arr, axis=0)
        
        # Interpolate intensities to 512 points to match the GUI shape
        if len(intensities_arr) != 512:
            x_old = np.linspace(0, 1, len(intensities_arr))
            x_new = np.linspace(0, 1, 512)
            intensities_512 = np.interp(x_new, x_old, intensities_arr)
        else:
            intensities_512 = intensities_arr
            
        # Return a 2D image matrix of shape (512, 512) where each row is intensities_512 / 512.
        # When GUI does np.sum(img, axis=0), it gets exactly intensities_512.
        img_row = intensities_512 / 512.0
        return np.tile(img_row, (512, 1))

    def get_idn(self) -> dict:
        return {
            'vendor': 'Princeton Instruments',
            'model': 'Remote LightField Camera Client',
            'serial': 'remote',
            'firmware': None
        }

    def abort_acquisition(self) -> None:
        """Abort the current remote spectrum acquisition."""
        try:
            if self.spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
                # Release the main proxy connection to close the active socket and free the thread
                if hasattr(self.remote_lf, "abort_acquisition"):
                    self.remote_lf.abort_acquisition()
                print("ClientLightFieldCamera: Released WinSpec main proxy connection to abort scan.")
            else:
                uri = getattr(self.remote_lf, '_pyroUri', None)
                if uri:
                    import Pyro4
                    with Pyro4.Proxy(uri) as abort_proxy:
                        abort_proxy._pyroTimeout = 2.0
                        abort_proxy.abort_acquisition()
                    print("ClientLightFieldCamera: Sent remote abort command via temporary proxy.")
        except Exception as e:
            print(f"ClientLightFieldCamera: Failed to send abort command: {e}")

class ClientLightFieldMonochromator(Instrument):
    """
    QCoDeS client driver for the Acton SpectraPro Spectrometer
    exposed by the unified LightField Pyro4 server.
    """
    def __init__(self, name, server_ip="G15-Pylon.dhcp-int.phys.ethz.ch", port=50920, pyro_uri=None, spectrometer_type="lightfieldG15", **kwargs):
        super().__init__(name, **kwargs)
        if spectrometer_type == 'lightfieldG11':
            self.remote_lf = NIRvanaMonoWrapper()
        else:
            Pyro4.config.SERIALIZER = 'pickle'
            Pyro4.config.PICKLE_PROTOCOL_VERSION = 2
            if pyro_uri is None:
                pyro_uri = f"PYRO:LightField@{server_ip}:{port}"
            proxy = Pyro4.Proxy(pyro_uri)
            if spectrometer_type in ['wingspecG15', 'winspecG11', 'wingspecG11', 'winspecG15']:
                self.remote_lf = PyroProxyWrapper(proxy)
            else:
                self.remote_lf = proxy
        
        self.add_parameter('wavelength',
                           label='Wavelength',
                           unit='nm',
                           get_cmd=lambda: remote_get(self.remote_lf, 'wavelength', 'get_wavelength'),
                           set_cmd=lambda val: remote_set(self.remote_lf, 'wavelength', val, 'set_wavelength'))

    def get_idn(self) -> dict:
        return {
            'vendor': 'Acton',
            'model': 'Remote LightField Spectrometer Client',
            'serial': 'remote',
            'firmware': None
        }