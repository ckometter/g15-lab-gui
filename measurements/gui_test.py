import os
import sys
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import h5py
import json

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from station_setup import initialize_station

class SMUSweepGUI:
    def __init__(self, root, station):
        self.root = root
        self.station = station
        self.root.title("SMU Channel 1 Voltage Sweep Controller")
        self.root.geometry("850x850")
        self.root.configure(bg="#1e1e2e")
        
        # Sweep state parameters
        self.is_running = False
        self.is_paused = False
        self.sweep_thread = None
        
        # Sweep data structure
        self.voltages_list = []
        self.current_index = 0
        self.x_data = []
        self.y_data = []
        
        # Load stored parameters or use defaults
        stored_values = self.load_from_value_storage()
        self.default_exposure = stored_values.get("exposure_time", 0.1)
        self.default_settling = stored_values.get("settling_time", 0.05)
        self.default_ramp_rate = stored_values.get("ramprate1", stored_values.get("ramp_rate", 0.1))
        self.default_ramp_step = stored_values.get("ramp_step", 0.0005)
        
        self.setup_ui()
        self.setup_plot()

    def load_from_value_storage(self):
        storage_path = os.path.join(parent_dir, "value_storage")
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    data = json.load(f)
                    # Migrate on read if necessary
                    if "ramprate1" not in data:
                        if "ramp_rate" in data:
                            data["ramprate1"] = data["ramp_rate"]
                        elif "ramp_rate_1" in data:
                            data["ramprate1"] = data["ramp_rate_1"]
                    return data
            except Exception as e:
                print(f"Error reading value storage: {e}")
        return {}

    def save_to_value_storage(self, exposure_time, settling_time, ramp_rate, ramp_step):
        storage_path = os.path.join(parent_dir, "value_storage")
        try:
            data = {}
            if os.path.exists(storage_path):
                with open(storage_path, "r") as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        pass
            data["exposure_time"] = exposure_time
            data["settling_time"] = settling_time
            data["ramprate1"] = ramp_rate
            data["ramprate2"] = ramp_rate
            data["ramp_step"] = ramp_step
            with open(storage_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error writing to value storage: {e}")

    def on_param_changed(self, event=None):
        try:
            exposure_time = float(self.exposure_entry.get())
            settling_time = float(self.settling_entry.get())
            ramp_rate = float(self.ramp_rate_entry.get())
            ramp_step = float(self.ramp_step_entry.get())
            self.save_to_value_storage(exposure_time, settling_time, ramp_rate, ramp_step)
            self.status_label.config(text="Status: Parameters saved to storage", fg="#a6e3a1")
        except ValueError:
            # If manual button click, show warning
            if event is None:
                messagebox.showerror("Invalid Input", "Please enter valid numeric parameters.")

    def on_param_changed_silent(self, event=None):
        try:
            exposure_time = float(self.exposure_entry.get())
            settling_time = float(self.settling_entry.get())
            ramp_rate = float(self.ramp_rate_entry.get())
            ramp_step = float(self.ramp_step_entry.get())
            self.save_to_value_storage(exposure_time, settling_time, ramp_rate, ramp_step)
        except ValueError:
            pass

    def setup_ui(self):
        # Left Panel (Controls)
        control_frame = tk.Frame(self.root, bg="#252538", width=250)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        control_frame.grid_propagate(False)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        
        title = tk.Label(
            control_frame, 
            text="SMU Ch1 Sweep", 
            font=("Arial", 14, "bold"), 
            bg="#252538", 
            fg="#cba6f7"
        )
        title.pack(pady=15)
        
        separator = tk.Frame(control_frame, height=2, bg="#313244")
        separator.pack(fill="x", padx=10, pady=5)
        
        # Start Voltage
        start_label = tk.Label(
            control_frame, 
            text="Start Voltage (V):", 
            font=("Arial", 10), 
            bg="#252538", 
            fg="#a6adc8"
        )
        start_label.pack(anchor="w", padx=15, pady=(15, 2))
        
        self.start_entry = tk.Entry(
            control_frame, 
            font=("Arial", 11), 
            bg="#313244", 
            fg="#cdd6f4", 
            insertbackground="#cdd6f4", 
            bd=0, 
            highlightthickness=1, 
            highlightbackground="#45475a", 
            highlightcolor="#b4befe"
        )
        self.start_entry.insert(0, "-0.005")
        self.start_entry.pack(fill="x", padx=15, pady=5)
        
        # Stop Voltage
        stop_label = tk.Label(
            control_frame, 
            text="Stop Voltage (V):", 
            font=("Arial", 10), 
            bg="#252538", 
            fg="#a6adc8"
        )
        stop_label.pack(anchor="w", padx=15, pady=(15, 2))
        
        self.stop_entry = tk.Entry(
            control_frame, 
            font=("Arial", 11), 
            bg="#313244", 
            fg="#cdd6f4", 
            insertbackground="#cdd6f4", 
            bd=0, 
            highlightthickness=1, 
            highlightbackground="#45475a", 
            highlightcolor="#b4befe"
        )
        self.stop_entry.insert(0, "0.005")
        self.stop_entry.pack(fill="x", padx=15, pady=5)
        
        # Steps
        steps_label = tk.Label(
            control_frame, 
            text="Number of Steps:", 
            font=("Arial", 10), 
            bg="#252538", 
            fg="#a6adc8"
        )
        steps_label.pack(anchor="w", padx=15, pady=(15, 2))
        
        self.steps_entry = tk.Entry(
            control_frame, 
            font=("Arial", 11), 
            bg="#313244", 
            fg="#cdd6f4", 
            insertbackground="#cdd6f4", 
            bd=0, 
            highlightthickness=1, 
            highlightbackground="#45475a", 
            highlightcolor="#b4befe"
        )
        self.steps_entry.insert(0, "21")
        self.steps_entry.pack(fill="x", padx=15, pady=5)
        
        # Exposure Time
        exposure_label = tk.Label(
            control_frame, 
            text="Exposure Time (s):", 
            font=("Arial", 10), 
            bg="#252538", 
            fg="#a6adc8"
        )
        exposure_label.pack(anchor="w", padx=15, pady=(10, 2))
        
        self.exposure_entry = tk.Spinbox(
            control_frame, 
            from_=0.001, to=10.0, increment=0.01,
            command=self.on_param_changed_silent,
            font=("Arial", 11), 
            bg="#313244", 
            fg="#cdd6f4", 
            insertbackground="#cdd6f4", 
            bd=0, 
            highlightthickness=1, 
            highlightbackground="#45475a", 
            highlightcolor="#b4befe"
        )
        self.exposure_entry.delete(0, tk.END)
        self.exposure_entry.insert(0, f"{self.default_exposure}")
        self.exposure_entry.pack(fill="x", padx=15, pady=5)
        
        # Settling Time
        settling_label = tk.Label(
            control_frame, 
            text="Settling Time (s):", 
            font=("Arial", 10), 
            bg="#252538", 
            fg="#a6adc8"
        )
        settling_label.pack(anchor="w", padx=15, pady=(10, 2))
        
        self.settling_entry = tk.Spinbox(
            control_frame, 
            from_=0.0, to=10.0, increment=0.01,
            command=self.on_param_changed_silent,
            font=("Arial", 11), 
            bg="#313244", 
            fg="#cdd6f4", 
            insertbackground="#cdd6f4", 
            bd=0, 
            highlightthickness=1, 
            highlightbackground="#45475a", 
            highlightcolor="#b4befe"
        )
        self.settling_entry.delete(0, tk.END)
        self.settling_entry.insert(0, f"{self.default_settling}")
        self.settling_entry.pack(fill="x", padx=15, pady=5)
        
        # Ramp Rate
        ramp_rate_label = tk.Label(
            control_frame, 
            text="Ramp Rate (V/s):", 
            font=("Arial", 10), 
            bg="#252538", 
            fg="#a6adc8"
        )
        ramp_rate_label.pack(anchor="w", padx=15, pady=(10, 2))
        
        self.ramp_rate_entry = tk.Spinbox(
            control_frame, 
            from_=0.001, to=100.0, increment=0.01,
            command=self.on_param_changed_silent,
            font=("Arial", 11), 
            bg="#313244", 
            fg="#cdd6f4", 
            insertbackground="#cdd6f4", 
            bd=0, 
            highlightthickness=1, 
            highlightbackground="#45475a", 
            highlightcolor="#b4befe"
        )
        self.ramp_rate_entry.delete(0, tk.END)
        self.ramp_rate_entry.insert(0, f"{self.default_ramp_rate}")
        self.ramp_rate_entry.pack(fill="x", padx=15, pady=5)

        # Ramp Step
        ramp_step_label = tk.Label(
            control_frame, 
            text="Ramp Step (V):", 
            font=("Arial", 10), 
            bg="#252538", 
            fg="#a6adc8"
        )
        ramp_step_label.pack(anchor="w", padx=15, pady=(10, 2))
        
        self.ramp_step_entry = tk.Spinbox(
            control_frame, 
            from_=0.0001, to=10.0, increment=0.0005,
            command=self.on_param_changed_silent,
            font=("Arial", 11), 
            bg="#313244", 
            fg="#cdd6f4", 
            insertbackground="#cdd6f4", 
            bd=0, 
            highlightthickness=1, 
            highlightbackground="#45475a", 
            highlightcolor="#b4befe"
        )
        self.ramp_step_entry.delete(0, tk.END)
        self.ramp_step_entry.insert(0, f"{self.default_ramp_step}")
        self.ramp_step_entry.pack(fill="x", padx=15, pady=5)

        # Bind events for automatic saving to value storage
        for entry in (self.exposure_entry, self.settling_entry, self.ramp_rate_entry, self.ramp_step_entry):
            entry.bind("<FocusOut>", self.on_param_changed)
            entry.bind("<Return>", self.on_param_changed)
            entry.bind("<KeyRelease>", self.on_param_changed_silent)

        # Save Parameters Button
        self.save_params_button = tk.Button(
            control_frame, 
            text="💾 Save Config to Storage", 
            command=self.on_param_changed, 
            bg="#f9e2af", 
            fg="#11111b", 
            activebackground="#b4befe", 
            font=("Arial", 9, "bold"), 
            bd=0, 
            cursor="hand2"
        )
        self.save_params_button.pack(fill="x", padx=15, pady=(5, 10))
        
        # Spacer
        tk.Frame(control_frame, height=20, bg="#252538").pack()
        
        # Start/Restart button
        self.start_button = tk.Button(
            control_frame, 
            text="▶ Start Sweep", 
            command=self.start_sweep, 
            bg="#a6e3a1", 
            fg="#11111b", 
            activebackground="#b4befe", 
            font=("Arial", 11, "bold"), 
            bd=0, 
            cursor="hand2"
        )
        self.start_button.pack(fill="x", padx=15, pady=8)
        
        # Pause button
        self.pause_button = tk.Button(
            control_frame, 
            text="⏸ Pause Sweep", 
            command=self.pause_sweep, 
            bg="#f38ba8", 
            fg="#11111b", 
            activebackground="#f9e2af", 
            font=("Arial", 11, "bold"), 
            bd=0, 
            state="disabled", 
            cursor="hand2"
        )
        self.pause_button.pack(fill="x", padx=15, pady=8)

        # Resume button
        self.resume_button = tk.Button(
            control_frame, 
            text="▶ Continue Sweep", 
            command=self.resume_sweep, 
            bg="#b4befe", 
            fg="#11111b", 
            activebackground="#f5e0dc", 
            font=("Arial", 11, "bold"), 
            bd=0, 
            state="disabled", 
            cursor="hand2"
        )
        self.resume_button.pack(fill="x", padx=15, pady=8)

        # Save button
        self.save_button = tk.Button(
            control_frame, 
            text="💾 Save Sweep Data", 
            command=self.save_data_dialog, 
            bg="#313244", 
            fg="#11111b", 
            activebackground="#b4befe", 
            font=("Arial", 11, "bold"), 
            bd=0, 
            state="disabled", 
            cursor="hand2"
        )
        self.save_button.pack(fill="x", padx=15, pady=8)
        
        # Status
        self.status_label = tk.Label(
            control_frame, 
            text="Status: Idle", 
            font=("Arial", 10, "italic"), 
            bg="#252538", 
            fg="#89b4fa"
        )
        self.status_label.pack(side="bottom", pady=20)

    def setup_plot(self):
        # Right Panel (Plot)
        plot_frame = tk.Frame(self.root, bg="#1e1e2e")
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        
        # Matplotlib Figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor("#1e1e2e")
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1e1e2e")
        self.ax.set_title("SMU Channel 1 I-V Sweep", color="#cdd6f4", fontsize=12, pad=10)
        self.ax.set_xlabel("Ch1 Voltage (V)", color="#a6adc8")
        self.ax.set_ylabel("Ch1 Current (A)", color="#a6adc8")
        
        # Style spines and ticks
        for spine in self.ax.spines.values():
            spine.set_color("#313244")
        self.ax.tick_params(colors="#a6adc8")
        self.ax.grid(True, linestyle="--", alpha=0.3, color="#45475a")
        
        # Line plot
        self.line, = self.ax.plot([], [], color="#cba6f7", linewidth=2.5, marker="o", markersize=4)
        
        # Canvas widget
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.draw()

    def save_data_dialog(self):
        """Open a standard save-as dialog and save the sweep data to the chosen HDF5 file."""
        if not self.x_data:
            messagebox.showwarning("No Data", "There is no sweep data to save yet.")
            return
            
        try:
            # Create data directory path
            data_dir = os.path.join(parent_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_filename = f"smu_sweep_{timestamp}.h5"
            
            # Open save-as file dialog
            filename = filedialog.asksaveasfilename(
                title="Save Sweep Data",
                defaultextension=".h5",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
                initialdir=data_dir,
                initialfile=default_filename
            )
            
            # If user canceled the dialog, return
            if not filename:
                return
                
            with h5py.File(filename, 'w') as f:
                # Store metadata as attributes
                f.attrs['start_voltage'] = float(self.start_entry.get())
                f.attrs['stop_voltage'] = float(self.stop_entry.get())
                f.attrs['num_steps'] = int(self.steps_entry.get())
                f.attrs['timestamp'] = timestamp
                f.attrs['sweep_status'] = "finished" if self.current_index >= len(self.voltages_list) else f"paused_at_{self.current_index}"
                
                # Store sweep data
                f.create_dataset('voltage', data=np.array(self.x_data))
                f.create_dataset('current', data=np.array(self.y_data))
                
            print(f"Manually saved sweep data to: {filename}")
            self.status_label.config(text=f"Saved data: {os.path.basename(filename)}", fg="#a6e3a1")
        except Exception as e:
            messagebox.showerror("HDF5 Save Error", f"Could not save sweep dataset:\n{e}")

    def start_sweep(self):
        if self.is_running:
            return
            
        # If we were previously paused, we can reset without warning but data won't autosave anymore
        self.is_paused = False
            
        try:
            start_v = float(self.start_entry.get())
            stop_v = float(self.stop_entry.get())
            steps = int(self.steps_entry.get())
            exposure_time = float(self.exposure_entry.get())
            settling_time = float(self.settling_entry.get())
            ramp_rate = float(self.ramp_rate_entry.get())
            ramp_step = float(self.ramp_step_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric parameters.")
            return
            
        if steps <= 1:
            messagebox.showerror("Invalid Input", "Steps must be greater than 1.")
            return

        # Save values to value_storage
        self.save_to_value_storage(exposure_time, settling_time, ramp_rate, ramp_step)

        # Configure button states
        self.start_button.config(state="disabled", bg="#313244")
        self.pause_button.config(state="normal", bg="#f38ba8")
        self.resume_button.config(state="disabled", bg="#313244")
        self.save_button.config(state="disabled", bg="#313244")
        self.status_label.config(text="Status: Sweeping Ch1...", fg="#f9e2af")
        
        # Reset and prepare dataset
        self.voltages_list = np.linspace(start_v, stop_v, steps)
        self.current_index = 0
        self.x_data = []
        self.y_data = []
        self.line.set_data(self.x_data, self.y_data)
        self.ax.set_xlim(min(start_v, stop_v) - 0.001, max(start_v, stop_v) + 0.001)
        self.canvas.draw()
        
        # Launch loop thread
        self.is_running = True
        self.sweep_thread = threading.Thread(
            target=self.run_sweep_loop, 
            daemon=True
        )
        self.sweep_thread.start()

    def resume_sweep(self):
        if self.is_running or not self.is_paused:
            return
            
        try:
            exposure_time = float(self.exposure_entry.get())
            settling_time = float(self.settling_entry.get())
            ramp_rate = float(self.ramp_rate_entry.get())
            ramp_step = float(self.ramp_step_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric parameters.")
            return

        # Save values to value_storage
        self.save_to_value_storage(exposure_time, settling_time, ramp_rate, ramp_step)
            
        self.is_running = True
        self.is_paused = False
        
        self.start_button.config(state="disabled", bg="#313244")
        self.pause_button.config(state="normal", bg="#f38ba8")
        self.resume_button.config(state="disabled", bg="#313244")
        self.save_button.config(state="disabled", bg="#313244")
        self.status_label.config(text="Status: Resuming Ch1...", fg="#f9e2af")
        
        self.sweep_thread = threading.Thread(
            target=self.run_sweep_loop, 
            daemon=True
        )
        self.sweep_thread.start()

    def run_sweep_loop(self):
        smu = self.station.smu
        total_steps = len(self.voltages_list)
        
        try:
            # Enable output
            smu.ch1.output(True)
            
            while self.current_index < total_steps and self.is_running:
                v = self.voltages_list[self.current_index]
                
                # 1. Apply voltage using the channel's ramped set helper
                smu.ch1.set_r.volt(v)
                
                # Load settling time dynamically from value storage during measuring
                stored_values = self.load_from_value_storage()
                settling_time = stored_values.get("settling_time", 0.0)
                
                # Wait the settling time (taken out of the value storage)
                if settling_time > 0:
                    time.sleep(settling_time)
                
                # 2. Read current
                curr = smu.ch1.measure_curr()
                
                # 3. Save coordinates
                self.x_data.append(v)
                self.y_data.append(curr)
                
                # Increment state step index
                self.current_index += 1
                
                # Update GUI (Thread-safe call)
                self.root.after(0, self.update_plot_ui, self.current_index, total_steps)
                
                time.sleep(0.05)
                
        except Exception as e:
            self.root.after(0, self.handle_error, str(e))
            
        finally:
            # Safely disable output when paused or done
            try:
                smu.ch1.output(False)
            except Exception:
                pass
                
        self.root.after(0, self.sweep_thread_finished)

    def update_plot_ui(self, step, total):
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw()
        self.status_label.config(text=f"Status: Point {step}/{total} (v={self.x_data[-1]:.4f} V)", fg="#f9e2af")

    def handle_error(self, err):
        self.is_running = False
        messagebox.showerror("Sweep Error", f"Error during sweep loop:\n{err}")

    def pause_sweep(self):
        if not self.is_running:
            return
        self.is_running = False
        self.is_paused = True
        self.status_label.config(text="Status: Pausing...", fg="#f38ba8")

    def sweep_thread_finished(self):
        self.is_running = False
        total_steps = len(self.voltages_list)
        
        # If we have collected any data, enable the save button
        if self.x_data:
            self.save_button.config(state="normal", bg="#fab387")  # Enable save (pastel orange)
        
        if self.current_index >= total_steps:
            # Sweep successfully completed
            self.is_paused = False
            self.status_label.config(text="Status: Finished (Data Ready to Save)", fg="#a6e3a1")
            self.start_button.config(state="normal", text="↺ Restart Sweep", bg="#a6e3a1")
            self.pause_button.config(state="disabled", bg="#313244")
            self.resume_button.config(state="disabled", bg="#313244")
        else:
            if self.is_paused:
                # Sweep was paused
                self.status_label.config(text=f"Status: Paused ({self.current_index}/{total_steps})", fg="#f9e2af")
                self.start_button.config(state="normal", text="↺ Restart Sweep", bg="#a6e3a1")
                self.pause_button.config(state="disabled", bg="#313244")
                self.resume_button.config(state="normal", bg="#b4befe")
            else:
                # Stopped/errored
                self.start_button.config(state="normal", text="▶ Start Sweep", bg="#a6e3a1")
                self.pause_button.config(state="disabled", bg="#313244")
                self.resume_button.config(state="disabled", bg="#313244")

def show_gui(station):
    """Launch the GUI from a Python shell or Jupyter notebook with the active station."""
    root = tk.Tk()
    app = SMUSweepGUI(root, station)
    root.mainloop()

if __name__ == '__main__':
    print("Initializing physical station for sweep GUI test...")
    simulated_devices = {
        'smu': False,          # Connect to the physical Keysight SMU
        'monochromator': True, 
        'rotator1': True,      
        'rotator2': True,      
        'camera': True,        
        'piezo150': True,      
        'piezo300': True       
    }
    station = initialize_station(simulated=False, simulated_devices=simulated_devices)
    root = tk.Tk()
    app = SMUSweepGUI(root, station)
    root.mainloop()
