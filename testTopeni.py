import numpy as np


def calculation_temperature_and_energy(upper_limit, lower_limit):
    w, l, h = 4, 4, 2.5  # Room dimensions (meters)
    T_init = lower_limit  # Initial air temperature (°C)
    heat_source_power = 800  # Power in W
    heat_source_efficiency = 0.85  # Efficiency in % (range 0-1)
    c = 1007  # Specific heat capacity in J/(kg*K)
    m = 1.225  # Air mass in kg/m³
    room_volume = w * l * h  # Room volume
    energy_consumption = 0

    delta_temperature_inside_outside = 19 - 17  # Temperature difference between inside and outside in °C

    time_hr = 1  # Time in hours
    delta_temperature = 0.5  # Temperature loss in one hour in °C
    heat_gain_rate = (heat_source_power * heat_source_efficiency) / (room_volume * m * c)  # Heat gain per second
    loss_coefficient = (delta_temperature * room_volume * m * c) / (2 * (w * l + l * h + h * w) * 3600 * delta_temperature_inside_outside) # Wall loss coefficient W/(m^2*K)
    temperature_outside = 0  # Outside temperature (°C)
    heat_lost = lambda in_temperature: (loss_coefficient * 2 * (w * l + l * h + h * w) * (temperature_outside - in_temperature)) / (room_volume * m * c)

    total_time = 3600 * 24 * 3 # Total simulation time (s)

    temperature = np.zeros(total_time)  # Initialize temperature array
    temperature[0] = T_init  # Set initial temperature
    boiler_on = False  # Initialize boiler state

    for i in range(0, total_time-1):
        delta_temperature = 0  # Temperature change

        if temperature[i] <= lower_limit:
            boiler_on = True

        if temperature[i] >= upper_limit:
            boiler_on = False

        heat_gain_rate_now = 0
        if boiler_on:
            heat_gain_rate_now = heat_gain_rate
            energy_consumption += heat_source_power * heat_source_efficiency / 3.6E6

        heat_lost_now = heat_lost(temperature[i])
        delta_temperature = heat_lost_now + heat_gain_rate_now
        temperature[i+1] = temperature[i] + delta_temperature
    return energy_consumption


values = np.arange(-3, 3, 0.5)
upper_limit = values + 21.5
lower_limit = values + 20.5
energy = np.zeros_like(values)

for idx, (up_lim, low_lim) in enumerate(zip(upper_limit, lower_limit)):
    energy[idx] = calculation_temperature_and_energy(up_lim, low_lim)


energy = energy / energy[len(energy) // 2]
price = 4E4
for e, low, up in zip(energy, lower_limit, upper_limit):
    print(f"Energy: {e:.3f}, Range: {low:.2f}:{up:.2f}, Price: {price * e:.1f}")
