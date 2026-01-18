import numpy as np
import matplotlib.pyplot as plt

# ===========================
#  GENERIC BATTERY SIMULATOR
# ===========================

class BatterySimulator:
    def __init__(self, capacity_ah=4.0):
        # --- Battery Parameters (1RC Model) ---
        self.Q = capacity_ah * 3600  # Capacity in Amp-seconds (Coulombs)
        self.R0 = 0.05               # Internal resistance that causes instant voltage drop (Ohms)
        self.R1 = 0.02               # Resistance that causes slow voltage effects (Ohms)
        self.C1 = 500                # Capacitance that stores temporary voltage effects (Farads)
        
        # --- Initial State ---
        self.soc = 1.0               # 100% full
        self.v_c = 0.0               # Initial polarization voltage is 0, starts with no extra internal voltage stored

    #Function that returns battery voltage when nothing is connected (Open Circuit Voltage) depends only on charge level    
    def get_ocv(self, soc):
        """
        Open Circuit Voltage curve approximation for a generic Li-ion cell.
        Use here a mathematical function to create the characteristic S-curve.
        """
        # This function approximates the OCV vs SoC curve
        return 3.5 + 0.7 * soc + 0.2 * np.log(soc + 1e-4) - 0.05 * np.log(1 - soc + 1e-4)

    #Simulation step that updates battery state based on current drawn and time step
    def step(self, current, dt):
        """
        current: Amps (positive for discharge)
        dt: time step in seconds
        """
        # 1. Update SoC (Coulomb Counting)
        # dSoC = -I / Q * dt
        self.soc = self.soc - (current * dt / self.Q) #Battery loses charge when current flows out
        
        # Makes sure SOC stays between 0% and 100%
        self.soc = np.clip(self.soc, 0, 1)

        # 2. Update Polarization Voltage (RC Circuit Dynamics)
        # Vc_new = Vc_old * exp(-dt/RC) + R1 * I * (1 - exp(-dt/RC))
        exp_val = np.exp(-dt / (self.R1 * self.C1)) #Calculates how fast internal voltage fades over time
        self.v_c = self.v_c * exp_val + (self.R1 * current * (1 - exp_val)) #Updates slow voltage effects caused by: Chemical delay and Internal resistance

        # 3. Calculate Terminal Voltage
        # V_term = OCV - I*R0 - Vc == [ideal voltage - instant drop - slow/internal effects]
        ocv = self.get_ocv(self.soc) #Gets voltage based on how full the battery is.
        voltage = ocv - (current * self.R0) - self.v_c 
        
        return voltage, self.soc, ocv

# --- Simulation Setup ---
dt = 1.0               # 1 second time steps
duration = 5400        # Simulate for 1.5 hour
time = np.arange(0, duration, dt)

# Create a "Drive Cycle" (Dynamic Current Profile)
# We simulate a load that changes every few minutes
current_profile = np.zeros_like(time)
# --- Cycle 1 ---
current_profile[200:800]   = 1.2    # Moderate discharge
current_profile[800:1100]  = 0.0    # Rest
current_profile[1100:1600] = 2.0    # Heavy discharge
current_profile[1600:1900] = 0.0    # Rest
# --- Charge 1 ---
current_profile[1900:2400] = -1.0   # Charging
current_profile[2400:2700] = 0.0    # Rest
# --- Cycle 2 ---
current_profile[2700:3300] = 1.5    # Moderate discharge
current_profile[3300:3600] = 0.0    # Rest
current_profile[3600:4100] = 2.5    # Heavy discharge
# --- Charge 2 ---
current_profile[4100:4600] = -1.2   # Charging
current_profile[4600:4900] = 0.0    # Rest
# --- Final light usage ---
current_profile[4900:5300] = 0.8    # Light discharge

# --- Run the Simulation ---
sim = BatterySimulator() #Create the battery
voltages = []
real_socs = []
noisy_voltages = []

print("Running simulation...")

for t, i in zip(time, current_profile):
    v, soc, ocv = sim.step(i, dt)   #Update battery for this second
    
    # Add Noise (Simulate a cheap sensor)
    # This is what our Kalman Filter will eventually have to deal with!
    noise = np.random.normal(0, 0.008) # Standard deviation of 8mV
    v_noisy = v + noise

    # Store everything for plotting
    voltages.append(v)
    noisy_voltages.append(v_noisy)
    real_socs.append(soc)

# =======================================
#  EXTENDED KALMAN FILTER IMPLEMENTATION
# =======================================

# --- Define the EKF Class ---
class ExtendedKalmanFilter:
    def __init__(self, capacity_ah=4, dt=1.0):
        # System Parameters (Must match the "Physics" roughly)
        self.Q = capacity_ah * 3600
        self.R0 = 0.05
        self.R1 = 0.02
        self.C1 = 500
        self.dt = dt
        
        # State Vector [SoC, Vc]
        # We start with a guess (e.g., we think battery is at 50%)
        self.x = np.array([[0.2], [0.0]]) 
        
        # Covariance Matrix (P) - How unsure we are
        # High numbers = "I have no idea where I am", I am very unsure about my estimate.
        self.P = np.array([[0.1, 0], [0, 0.1]])
        
        # Process Noise Matrix (Q_cov) - Uncertainty in the physics model
        # (e.g., maybe our capacity value is slightly wrong)
        self.Q_cov = np.array([[1e-7, 0], [0, 1e-5]])
        
        # Measurement Noise Matrix (R_cov) - Uncertainty in the sensor
        # (Based on the noise we added in the simulator: 0.008V)
        self.R_cov = np.array([[0.008**2]])

    def get_ocv_slope(self, soc):
        """
        Calculates d(OCV)/d(SoC).
        This is the derivative of the OCV function we used in the simulator.
        Formula: 0.7 + 0.2/(soc) + 0.05/(1-soc)
        """
        # Avoid division by zero
        soc = np.clip(soc, 0.001, 0.999) 
        return 0.7 + (0.2 / (soc + 1e-4)) + (0.05 / (1 - soc + 1e-4))

    def get_ocv(self, soc):
        # Same OCV curve as the simulator
        return 3.5 + 0.7 * soc + 0.2 * np.log(soc + 1e-4) - 0.05 * np.log(1 - soc + 1e-4)

    def predict(self, current):
        """
        STEP 1: PREDICT (Time Update)
        Estimate where the system is based on physics equations.
        """
        # A Matrix (State Transition)
        # SoC_new = SoC_old
        # Vc_new = Vc_old * exp(-dt/RC)
        exp_val = np.exp(-self.dt / (self.R1 * self.C1)) #How fast internal voltage fades
        A = np.array([[1, 0], [0, exp_val]]) #SoC stays mostly the same, Vc decays slowly
        
        # B Matrix (Control Input)
        # Effect of Current on SoC and Vc
        B = np.array([[-self.dt / self.Q], [self.R1 * (1 - exp_val)]])
        
        # 1. Extrapolate the State
        # x = A*x + B*u
        self.x = A @ self.x + B * current #EKF guesses new SoC & Vc using physics only
        
        # 2. Extrapolate Uncertainty
        # P = A*P*A^T + Q
        self.P = A @ self.P @ A.T + self.Q_cov #Uncertainty increases over time, we are less sure about our estimate now

    def update(self, voltage_measured, current):
        """
        STEP 2: UPDATE (Measurement Correction)
        Correct the prediction using the actual voltage reading.
        """
        soc_pred = self.x[0, 0]
        vc_pred = self.x[1, 0]
        
        # Expected Voltage (h(x))
        # V = OCV(SoC) - Vc - I*R0
        voltage_pred = self.get_ocv(soc_pred) - vc_pred - (current * self.R0)
        
        # Calculate Jacobian (H Matrix) - Linearize the curve
        d_ocv = self.get_ocv_slope(soc_pred)
        
        # H = [dVoltage/dSoC, dVoltage/dVc]
        # H = [dOCV/dSoC, -1]
        H = np.array([[d_ocv, -1]])
        
        # Calculate Kalman Gain (K)
        # K = P*H^T * (H*P*H^T + R)^-1
        S = H @ self.P @ H.T + self.R_cov # Total expected error
        K = self.P @ H.T @ np.linalg.inv(S) # How much we trust the measurement vs our prediction
        
        # Update State Estimate
        # x = x + K * (y - y_expected)
        y_residual = voltage_measured - voltage_pred
        self.x = self.x + K * y_residual #EKF corrects itself
        self.x[0, 0] = np.clip(self.x[0, 0], 0.001, 0.999)
        
        # Update Uncertainty
        # P = (I - K*H) * P
        I = np.eye(2)
        self.P = (I - K @ H) @ self.P
        
        return self.x[0, 0] # Return the estimated SoC

# --- Run the Comparison Loop ---

# Initialize
ekf = ExtendedKalmanFilter(dt=dt)
ekf_socs = []

# Simple Coulomb Counter for comparison (The "Dumb" method)
cc_soc = 0.7 # Start with same wrong guess
cc_socs = []

print("Running EKF Estimation...")

for i, v_noisy in zip(current_profile, noisy_voltages):
    
    # -- Run Coulomb Counter (for reference) --
    # Just integrate current, no correction
    cc_soc = cc_soc - (i * dt / ekf.Q)
    cc_socs.append(cc_soc)
    
    # -- Run Extended Kalman Filter --
    ekf.predict(i)
    est_soc = ekf.update(v_noisy, i)
    ekf_socs.append(est_soc)

# ==================
#   VISUALIZATION
# ==================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Top Plot: Current
ax1.plot(time, current_profile, color='blue')
ax1.set_ylabel("Current (Amps)")
ax1.set_title("Input: Dynamic Current Profile")
ax1.grid(True)

# Middle Plot: Voltage
ax2.plot(time, voltages, color='red', label='True Voltage', linewidth=2)
ax2.plot(time, noisy_voltages, color='black', alpha=0.3, label='Sensor Data (Noisy)')
ax2.set_ylabel("Voltage (Volts)")
ax2.set_xlabel("Time (seconds)")
ax2.set_title("Output: Battery Voltage Response")
ax2.legend()
ax2.grid(True)

# Bottom Plot: SoC Estimates
ax3.plot(time, real_socs, label='True SoC', color='green', linewidth=2)
ax3.plot(time, ekf_socs, label='EKF Estimate', color='blue', linestyle='--')
ax3.plot(time, cc_socs, label='Coulomb Counting', color='orange', linestyle=':')
ax3.set_ylabel("State of Charge (SoC)")
ax3.set_xlabel("Time (seconds)")
ax3.set_title("EKF vs. Coulomb Counting vs. Truth")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()