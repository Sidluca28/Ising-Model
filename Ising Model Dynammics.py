import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

size = 50
J = 1
temperature = 1

# Initialize spins randomly
spins = np.random.choice([-1, 1], size=(size, size))

# Create figure and axes for visualization
fig, (ax, ax_magnetization) = plt.subplots(1, 2, figsize=(12, 8))
title_text = ax.set_title('Ising Model Simulation (Temperature: {:.2f})'.format(temperature), fontsize=16)
ax.set_xticks([])
ax.set_yticks([])

# Set the initial plot with a more visually appealing colormap
image = ax.imshow(spins, vmin=-1, vmax=1, cmap='coolwarm')
cbar = plt.colorbar(image, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Spin', rotation=270, labelpad=15)

# Set up the magnetization plot
ax_magnetization.set_xlabel('Temperature')
ax_magnetization.set_ylabel('Magnetization')
line, = ax_magnetization.plot([], [], color='blue', lw=2)

# Define the slider position and size
slider_ax = plt.axes([0.2, 0.02, 0.6, 0.03])

# Create the temperature slider with a customized appearance
temperature_slider = Slider(slider_ax, 'Temperature\n(in terms of kbT)', 0.01, 4, valinit=temperature, color='skyblue')

# Define the button position and size
button_ax = plt.axes([0.85, 0.02, 0.1, 0.03])

# Create the pause/play button
button = Button(button_ax, 'Pause')

# Variables for simulating slider movement and animation state
slider_min = temperature_slider.valmin
slider_max = temperature_slider.valmax
slider_direction = 1  # 1 for increasing, -1 for decreasing
slider_speed = 0.01  # Adjust the speed of slider movement
pause_animation = False

# Lists to store data for magnetization plot
temperatures = []
magnetizations = []


# Function to update the plot at each frame
def update(frame):
    global spins, temperature, slider_direction

    if not pause_animation:
        # Move the slider value
        temperature += slider_direction * slider_speed

        # Reverse the direction at the limits
        if temperature >= slider_max or temperature <= slider_min:
            slider_direction *= -1

        # Perform a Monte Carlo update
        magnetization = np.sum(spins)
        temperatures.append(temperature)
        magnetizations.append(magnetization)

        for _ in range(size ** 2):
            i = np.random.randint(size)
            j = np.random.randint(size)

            # Calculate the energy change
            delta_E = 2 * J * spins[i, j] * (
                    spins[(i + 1) % size, j] + spins[(i - 1) % size, j] + spins[i, (j + 1) % size] + spins[
                i, (j - 1) % size]
            )

            # Accept or reject the spin flip
            if delta_E < 0 or np.exp(-delta_E / temperature) > np.random.rand():
                spins[i, j] *= -1

        # Update the plots
        image.set_array(spins)
        title_text.set_text('Ising Model Simulation (Temperature: {:.2f})'.format(temperature))
        line.set_data(temperatures, magnetizations)
        ax_magnetization.relim()
        ax_magnetization.autoscale_view()


# Function to handle button click (pause/play)
def on_button_click(event):
    global pause_animation
    pause_animation = not pause_animation
    button.label.set_text('Play' if pause_animation else 'Pause')


# Function to handle slider movement
def on_slider_change(val):
    global temperature
    temperature = temperature_slider.val


# Connect button click event
button.on_clicked(on_button_click)

# Connect slider change event
temperature_slider.on_changed(on_slider_change)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=False)

# Show the plot
plt.show()
