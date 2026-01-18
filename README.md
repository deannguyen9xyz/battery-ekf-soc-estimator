# battery-ekf-soc-estimator
A Python-based Extended Kalman Filter (EKF) implementation for Li-ion battery State of Charge (SoC) estimation using a 1RC Equivalent Circuit Model

---

## 🎯 Purpose of This Project

Purpose of This Project The goal of this project is to solve the two biggest problems in Battery Management Systems (BMS):

***Sensor Noise**: Real-world voltage sensors are "noisy" and jumpy, making direct measurements unreliable.

***Estimation Drift**: Standard methods like "Coulomb Counting" (simply adding up current) accumulate errors over time and cannot correct themselves if they start with a wrong guess.

This project implements an Extended Kalman Filter (EKF) to create a "Digital Twin" of a battery. By combining a physical model (1RC Equivalent Circuit) with real-time sensor data, the algorithm can "see through the noise" and self-correct its State of Charge (SoC) estimate.

--- 

## ▶️ How to Run

1. Install dependencies:

       pip install numpy matplotlib

3. Run the script: **python battery_ekf_soc.py**

---

## 📊 Result and Discussion

<img width="1107" height="786" alt="image" src="https://github.com/user-attachments/assets/8b875319-71ae-4bd6-851c-e4d61b424263" />

1. **Input: Dynamic Current Profile** (Top Plot)

The simulation uses a varying current profile to mimic real-world usage, such as an electric vehicle drive cycle.

-Positive Current: Represents the battery discharging (powering a motor).

-Negative Current: Represents regenerative charging, where energy is pushed back into the cell.

-Zero Current: Represents "rest periods" to observe the battery's chemical relaxation.

2. **Output: Battery Voltage Response** (Middle Plot)

This plot showcases the physical behavior of the 1RC Equivalent Circuit Model:

-Ohmic Drop: The sharp vertical jumps in the red line occur the moment current changes, representing internal resistance ($R_0$).

-Polarization/Diffusion: The gradual curves after a current change represent the RC dynamics ($R_1, C_1$), simulating the time it takes for ions to redistribute.

-Sensor Noise (Grey Line): To simulate real hardware, Gaussian noise was added to the "True Voltage." This "fuzz" is the only data the EKF "sees" to make its corrections.

3. **EKF vs. Coulomb Counting vs. Truth** (Bottom Plot)

This is the core "brain" of the project. To test the robustness of the algorithm, we intentionally started both estimators with a significant error (Initial guess = 0% SoC, while the True battery = 100% SoC).

-True SoC (Green): The actual state of the battery based on physics.

-Coulomb Counting (Orange): This "blind" method simply integrates current. Because it started with a wrong guess, it remains wrong for the entire duration. It has no mechanism to correct itself using voltage.

-EKF Estimate (Blue Dashed): This is the Sensor Fusion result. Notice how the blue line rapidly converges to the green line within the first few minutes. By comparing the noisy voltage to its internal model, the EKF realizes its initial guess was wrong and "snaps" to the truth.

--- 

## 🧑‍💻 Author

Developed by: Vu Bao Chau Nguyen, Ph.D.

Keywords: Lithium ion battery, Kalman filter, State of charge.

---
