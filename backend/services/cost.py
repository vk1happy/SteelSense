def estimate_cost(energy_kwh, tariff):
    energy = float(energy_kwh)
    rate = float(tariff)
    if energy < 0:
        raise ValueError("Energy cannot be negative.")
    if rate <= 0:
        raise ValueError("Tariff must be greater than zero.")
    return energy * rate
