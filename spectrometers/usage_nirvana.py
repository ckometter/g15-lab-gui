import NIRvana_send_trigger as trig




exp_time=0.1 #seconds
num_frames = 2
spec_params = {"command": "start_acquisition", "exposure_time": exp_time*1e3, "num_frames": num_frames}  # exposure time is in ms 

lamb, spec = trig.trigger(spec_params)  # wls will be 640 x 1 array of wavlengths (nm), spec is num_frames x 640 array of counts
        


# check on RD desktop-lsqkit5.dhcp-phys.ethz.ch with pw 'photon1'