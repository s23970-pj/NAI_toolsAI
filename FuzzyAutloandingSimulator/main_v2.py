'''

Na początku kodu dodaj opis problemu; wymień autorów rozwiązania; dodaj instrukcję przygotowania środowisk.

AUTOR: ADRIAN GOIK
OPIS PROBLEMU: Symulator automatycznego lądowania 2D przy użyciu logiki rozmytej (fuzzy logic) implementujący zmiany
w czasie rzeczywistym bazujące na wysokości, Ground Speed (prędkości postępowej względem terenu) oraz odległości
strefy przyziemienia.

PRZYGOTOWANIE ŚRODOWISKA:
1. Pobrać kod źródłowy main
2. Zainstalować biblioteki numpy, skfuzzy, pygame oraz packaging
pip install packaging
pip install networkx

3. Uruchomić program

'''

import numpy as np
import skfuzzy as fuzz
import pygame
from skfuzzy import control as ctrl

# Initialize pygame
pygame.init()

# Screen settings
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Automatic Plane Landing Simulation")

# Colors for refe
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)

# Define the fuzzy input and output variables\Antecendants are inputs to the fuzy system which provide information about plane condition
altitude = ctrl.Antecedent(np.arange(0, 3001, 1), 'altitude')  # Altitude in feet range 0-3000
ground_speed = ctrl.Antecedent(np.arange(60, 181, 1), 'ground_speed')  # Ground speed in knots
distance_to_runway = ctrl.Antecedent(np.arange(0, 6.1, 0.1), 'distance_to_runway')  # Distance in nautical miles

rate_of_descent = ctrl.Consequent(np.arange(0, 1501, 1), 'rate_of_descent')  # Target descent rate in fpm

# Define membership functions based on Rules of Thumb for 3-degree glideslope
altitude['high'] = fuzz.trimf(altitude.universe, [1000, 3000, 3000]) # high altitudes at the beggining of approach
altitude['medium'] = fuzz.trimf(altitude.universe, [500, 1500, 2500]) # medium at the intermediate approach
altitude['low'] = fuzz.trimf(altitude.universe, [0, 0, 800]) # low at final aproach close to FLARE
'''
Rules of thumb in aviation are fast methods to calculate desired value. In this case we can use rule of thumb in calculating
3 degree glide slope. Altitude = Distance * 300 feet. We can call it target altitude
'''
ground_speed['slow'] = fuzz.trimf(ground_speed.universe, [60, 60, 100])
ground_speed['medium'] = fuzz.trimf(ground_speed.universe, [80, 120, 160])
ground_speed['fast'] = fuzz.trimf(ground_speed.universe, [140, 180, 180])

distance_to_runway['far'] = fuzz.trimf(distance_to_runway.universe, [2, 6, 6])
distance_to_runway['medium'] = fuzz.trimf(distance_to_runway.universe, [1, 3, 5])
distance_to_runway['close'] = fuzz.trimf(distance_to_runway.universe, [0, 0, 1])

# Rate of Descent Membership Functions
rate_of_descent['shallow'] = fuzz.trimf(rate_of_descent.universe, [0, 300, 600])
rate_of_descent['moderate'] = fuzz.trimf(rate_of_descent.universe, [400, 700, 1000])
rate_of_descent['steep'] = fuzz.trimf(rate_of_descent.universe, [800, 1200, 1500])
'''
Rate of descent rule of thumb is RoD = Ground Speed * 5
'''
# Define fuzzy rules to ensure continued descent
rule1 = ctrl.Rule(distance_to_runway['far'] & ground_speed['fast'] & altitude['high'], rate_of_descent['steep'])
rule2 = ctrl.Rule(distance_to_runway['medium'] & ground_speed['medium'] & altitude['medium'], rate_of_descent['moderate'])
rule3 = ctrl.Rule(distance_to_runway['close'] & ground_speed['slow'] & altitude['low'], rate_of_descent['shallow'])

# Additional rules for continued descent as plane gets close to runway
rule4 = ctrl.Rule(altitude['high'] & distance_to_runway['close'], rate_of_descent['steep'])
rule5 = ctrl.Rule(altitude['low'] & distance_to_runway['medium'], rate_of_descent['moderate'])
rule6 = ctrl.Rule(altitude['low'] & distance_to_runway['close'], rate_of_descent['shallow'])

# Create the control system and simulation
landing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
landing_simulation = ctrl.ControlSystemSimulation(landing_ctrl)

# Simulation initial parameters
plane_x = 100  # Starting position of the plane on screen
plane_altitude = 1800  # Initial altitude in feet
initial_ground_speed = 120  # Initial ground speed in knots
distance_to_touchdown = 6  # Initial distance to runway in nautical miles

# Main loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get(): # Process events in pygame event queue
        if event.type == pygame.QUIT: #checking if the window button had been pressed
            running = False

    # Calculate target altitude for 3-degree glideslope with Rule of Thumbs
    target_altitude = distance_to_touchdown * 300
    altitude_error = plane_altitude - target_altitude

    # Use fuzzy control to adjust rate of descent based on current altitude, speed, and distance to runway
    landing_simulation.input['altitude'] = plane_altitude
    landing_simulation.input['ground_speed'] = initial_ground_speed
    landing_simulation.input['distance_to_runway'] = distance_to_touchdown

    # Run fuzzy logic to compute the rate of descent
    try:
        landing_simulation.compute()
        rate_of_descent_output = landing_simulation.output.get('rate_of_descent', 300)
    except KeyError as e:  #exception handling when fuzzy does not generate output default descent rate is 300 fpm (enough to survive)
        print(f"Warning: Missing output {e}, using default values")
        rate_of_descent_output = 300  # Use a default shallow descent rate if no output

    # Adjust altitude based on rate of descent and altitude error
    if altitude_error > 0:
        # If above the glideslope, increase rate of descent
        actual_rate_of_descent = rate_of_descent_output + altitude_error * 0.1
    elif altitude_error < 0:
        # If below the glideslope, decrease rate of descent
        actual_rate_of_descent = max(rate_of_descent_output - abs(altitude_error * 0.1), 0)
    else:
        actual_rate_of_descent = rate_of_descent_output

    # Update altitude and distance
    plane_altitude -= actual_rate_of_descent / 60  # Convert fpm to fps
    distance_to_touchdown -= initial_ground_speed / 3600  # Convert knots to nm per second

    # Ensure descent continues to runway level
    if distance_to_touchdown < 0.5 and plane_altitude > 50:
        # Ensure descent rate to avoid level-off too early
        actual_rate_of_descent = max(actual_rate_of_descent, 500)  # Keep descent active near runway

    # Update plane position on screen based on descent path
    plane_x += 3  # Move plane horizontally
    plane_y = screen_height - int((plane_altitude / 1800) * screen_height)  # Map altitude to screen

    # Clear the screen
    screen.fill(WHITE)

    # Draw the plane
    pygame.draw.rect(screen, BLUE, (plane_x, plane_y, 40, 20))

    # Draw runway
    runway_width, runway_height = 200, 20
    runway_x = screen_width - runway_width - 50
    runway_y = screen_height - runway_height
    pygame.draw.rect(screen, GRAY, (runway_x, runway_y, runway_width, runway_height))

    # Display altitude, distance, and rate of descent
    font = pygame.font.Font(None, 36)
    altitude_text = font.render(f'Altitude: {int(plane_altitude)} ft', True, BLACK)
    descent_text = font.render(f'Distance to Runway: {distance_to_touchdown:.1f} nm', True, BLACK)
    rate_text = font.render(f'Rate of Descent: {int(actual_rate_of_descent)} fpm', True, BLACK)
    screen.blit(altitude_text, (10, 10))
    screen.blit(descent_text, (10, 50))
    screen.blit(rate_text, (10, 90))

    # Check if plane has reached the runway
    if distance_to_touchdown <= 0 or plane_y >= runway_y:
        landing_text = font.render('Landed!', True, (0, 128, 0))
        screen.blit(landing_text, (screen_width // 2 - 50, screen_height // 2))
        pygame.display.flip()
        pygame.time.wait(2000)
        running = False

    # Update display
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
