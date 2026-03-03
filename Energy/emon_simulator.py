import json
import time
import yaml
from random import uniform
from concurrent.futures import ThreadPoolExecutor

VOLTAGE_RANGE = (210, 230)

def read_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def simulate_device(device, details, frequency):
    count = details['count']
    power = details['power']
    voltage = round(uniform(*VOLTAGE_RANGE), 2)
    current = round(power / voltage, 2)
    energy = round((power * frequency / 3600) * count, 4)
    
    return {
        "device": device,
        "count": count,
        "voltage": voltage,
        "current": current,
        "energy_wh": energy
    }

def generate_single_reading(devices, frequency):
    aggregate_energy = 0
    device_readings = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(simulate_device, device, details, frequency) for device, details in devices.items()]
        for future in futures:
            reading = future.result()
            aggregate_energy += reading['energy_wh'] * reading['count']
            device_readings.append(reading)

    return {
        "timestamp": time.time(),
        "devices": device_readings,
        "aggregate_energy_wh": round(aggregate_energy, 4)
    }

def generate_readings(devices, num_readings, frequency):
    readings = []
    for _ in range(num_readings):
        reading = generate_single_reading(devices, frequency)
        readings.append(reading)
        print(reading['devices'])
        time.sleep(frequency)
    return readings

def output_readings_to_file(readings, output_file):
    with open(output_file, 'w') as file:
        json.dump(readings, file, indent=4)
    print("--" * 10)
    print(f"Readings written to {output_file}")

def main():
    config_path = "config.yaml"
    config = read_config(config_path)

    devices = config['devices']
    num_readings = config['num_readings']
    frequency = config['frequency_seconds']
    output_file = config['output_file']

    print("--" * 10)
    print(f"Will generate {num_readings} readings at {frequency} second intervals and write to file {output_file}")
    print("--" * 10)

    readings = generate_readings(devices, num_readings, frequency)
    output_readings_to_file(readings, output_file)

if __name__ == "__main__":
    main()
