import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# Physical constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# Body properties
class Body:
    def __init__(self, name, mass, position, velocity, color, size):
        self.name = name
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.size = size
        self.trajectory = [self.position.copy()]
    
    def update(self, acceleration, dt):
        """Update position and velocity using Verlet integration"""
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())

def gravitational_acceleration(body, other_bodies):
    """Calculate gravitational acceleration on a body from all other bodies"""
    acceleration = np.zeros(2)
    for other in other_bodies:
        if other is not body:
            r_vec = other.position - body.position
            r_mag = np.linalg.norm(r_vec)
            if r_mag > 0:
                acceleration += G * other.mass * r_vec / (r_mag ** 3)
    return acceleration

def simulate(bodies, dt, total_time, perturbation=None):
    """Simulate the n-body system"""
    num_steps = int(total_time / dt)
    
    for step in range(num_steps):
        current_time = step * dt
        
        # Apply perturbation if specified
        if perturbation and abs(current_time - perturbation['time']) < dt:
            body_idx = perturbation['body_index']
            bodies[body_idx].velocity += np.array(perturbation['delta_v'])
            print(f"Applied perturbation to {bodies[body_idx].name} at t={current_time:.2e} s")
        
        # Calculate accelerations for all bodies
        accelerations = [gravitational_acceleration(body, bodies) for body in bodies]
        
        # Update all bodies
        for body, acc in zip(bodies, accelerations):
            body.update(acc, dt)
    
    return bodies

def plot_trajectories(bodies, title="Orbital Trajectories"):
    """Plot the complete trajectories"""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    for body in bodies:
        trajectory = np.array(body.trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], '-', 
               color=body.color, alpha=0.6, linewidth=1.5, label=body.name)
        # Mark initial and final positions
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'o', 
               color=body.color, markersize=8, markeredgecolor='black', markeredgewidth=1)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's', 
               color=body.color, markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    ax.legend(loc='upper right', fontsize=10)
    
    # Add custom legend for markers
    circle_patch = mpatches.Patch(color='gray', label='Initial position')
    square_patch = mpatches.Patch(color='gray', label='Final position')
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [circle_patch, square_patch], 
             loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig

def setup_stable_system():
    """Set up a stable Sun-Earth-Moon system with realistic parameters"""
    # Sun
    sun_mass = 1.989e30  # kg
    sun_pos = [0, 0]
    sun_vel = [0, 0]
    
    # Earth (at 1 AU from Sun)
    earth_mass = 5.972e24  # kg
    earth_distance = 1.496e11  # 1 AU in meters
    earth_orbital_velocity = np.sqrt(G * sun_mass / earth_distance)
    earth_pos = [earth_distance, 0]
    earth_vel = [0, earth_orbital_velocity]
    
    # Moon (at ~384,400 km from Earth)
    moon_mass = 7.342e22  # kg
    moon_distance = 3.844e8  # meters from Earth
    moon_orbital_velocity_relative = np.sqrt(G * earth_mass / moon_distance)
    moon_pos = [earth_distance + moon_distance, 0]
    moon_vel = [0, earth_orbital_velocity + moon_orbital_velocity_relative]
    
    sun = Body("Sun", sun_mass, sun_pos, sun_vel, 'yellow', 15)
    earth = Body("Earth", earth_mass, earth_pos, earth_vel, 'blue', 10)
    moon = Body("Moon", moon_mass, moon_pos, moon_vel, 'gray', 5)
    
    return [sun, earth, moon]

def setup_perturbed_system():
    """Set up system identical to stable"""
    return setup_stable_system()

if __name__ == "__main__":
    print("3-Body Gravitational Simulation: Sun-Earth-Moon System")
    print("=" * 60)
    
    # Simulation parameters (optimized for speed)
    days_to_simulate = 90  # 3 months
    total_time = days_to_simulate * 24 * 3600
    dt = 7200  # Time step: 2 hours
    
    print(f"\nSimulation parameters:")
    print(f"  Duration: {days_to_simulate} days")
    print(f"  Time step: {dt} seconds ({dt/3600} hours)")
    print(f"  Total steps: {int(total_time/dt)}")
    
    # SCENARIO 1: Stable orbital system
    print("\n" + "=" * 60)
    print("SCENARIO 1: Stable System (No Perturbations)")
    print("=" * 60)
    
    bodies_stable = setup_stable_system()
    print("\nRunning simulation...")
    simulate(bodies_stable, dt, total_time)
    
    print("Creating trajectory plot...")
    fig1 = plot_trajectories(bodies_stable, "Stable Sun-Earth-Moon System (90 Days)")
    plt.savefig('stable_orbits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: stable_orbits.png")
    
    # SCENARIO 2: System with gravitational perturbation
    print("\n" + "=" * 60)
    print("SCENARIO 2: System with Perturbation")
    print("=" * 60)
    
    bodies_perturbed = setup_perturbed_system()
    
    # Add a velocity perturbation to Earth at day 30
    perturbation = {
        'body_index': 1,  # Earth
        'time': 30 * 24 * 3600,  # 30 days in seconds
        'delta_v': [0, 2000]  # Add 2 km/s in y-direction
    }
    
    print(f"\nPerturbation details:")
    print(f"  Target: {bodies_perturbed[perturbation['body_index']].name}")
    print(f"  Time: {perturbation['time']/(24*3600)} days")
    print(f"  Velocity change: {perturbation['delta_v']} m/s")
    
    print("\nRunning simulation...")
    simulate(bodies_perturbed, dt, total_time, perturbation=perturbation)
    
    print("Creating trajectory plot...")
    fig2 = plot_trajectories(bodies_perturbed, 
                             "Perturbed System: Earth Velocity Kick at Day 30")
    plt.savefig('perturbed_orbits.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: perturbed_orbits.png")
    
    # SCENARIO 3: Comparison plot
    print("\n" + "=" * 60)
    print("Creating comparison plot...")
    
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Stable system
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_title('Stable System', fontsize=14, fontweight='bold')
    
    for body in bodies_stable:
        trajectory = np.array(body.trajectory)
        ax1.plot(trajectory[:, 0], trajectory[:, 1], '-', 
                color=body.color, alpha=0.6, linewidth=1.5, label=body.name)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Perturbed system
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Y Position (m)', fontsize=12)
    ax2.set_title('Perturbed System (Earth kick at day 30)', fontsize=14, fontweight='bold')
    
    for body in bodies_perturbed:
        trajectory = np.array(body.trajectory)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], '-', 
                color=body.color, alpha=0.6, linewidth=1.5, label=body.name)
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: comparison.png")
    
    # Create animation for stable system (downsampled for speed)
    print("\n" + "=" * 60)
    print("Creating animation...")
    
    # Downsample trajectory for animation
    downsample_factor = 5
    
    fig_anim, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Sun-Earth-Moon System Animation', fontsize=14, fontweight='bold')
    
    # Calculate bounds
    all_x, all_y = [], []
    for body in bodies_stable:
        trajectory = np.array(body.trajectory)
        all_x.extend(trajectory[:, 0])
        all_y.extend(trajectory[:, 1])
    
    margin = 0.1
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    ax.set_xlim(min(all_x) - margin * x_range, max(all_x) + margin * x_range)
    ax.set_ylim(min(all_y) - margin * y_range, max(all_y) + margin * y_range)
    
    # Initialize plot elements
    trails = []
    points = []
    
    for body in bodies_stable:
        trail, = ax.plot([], [], '-', color=body.color, alpha=0.6, linewidth=1)
        point, = ax.plot([], [], 'o', color=body.color, markersize=body.size, 
                        label=body.name, markeredgecolor='white', markeredgewidth=0.5)
        trails.append(trail)
        points.append(point)
    
    ax.legend(loc='upper right', fontsize=10)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    trail_length = 100
    
    def animate(frame):
        actual_frame = frame * downsample_factor
        for i, (body, trail, point) in enumerate(zip(bodies_stable, trails, points)):
            trajectory = np.array(body.trajectory)
            start_idx = max(0, actual_frame - trail_length)
            trail.set_data(trajectory[start_idx:actual_frame, 0], 
                          trajectory[start_idx:actual_frame, 1])
            
            if actual_frame < len(trajectory):
                point.set_data([trajectory[actual_frame, 0]], [trajectory[actual_frame, 1]])
        
        current_time = actual_frame * dt
        days = current_time / (24 * 3600)
        time_text.set_text(f'Time: {days:.1f} days')
        
        return trails + points + [time_text]
    
    total_frames = len(bodies_stable[0].trajectory) // downsample_factor
    
    anim = FuncAnimation(fig_anim, animate, frames=total_frames,
                        interval=50, blit=True, repeat=True)
    
    anim.save('orbit_animation.gif', writer='pillow', fps=20, dpi=80)
    plt.close()
    print("  Saved: orbit_animation.gif")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("\nGenerated files:")
    print("  1. stable_orbits.png - Stable system trajectories")
    print("  2. perturbed_orbits.png - Perturbed system trajectories")
    print("  3. comparison.png - Side-by-side comparison")
    print("  4. orbit_animation.gif - Animated orbits")
    print("=" * 60)
