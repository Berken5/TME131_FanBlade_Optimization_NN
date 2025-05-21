import re
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

class FluidGridInterpolator:
    def __init__(self, simulation_file, all_nodes_file):
        self.simulation_file = simulation_file
        self.all_nodes_file = all_nodes_file
        self.fields = ["Pressure (Pa)", "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)"]
        self.params = self._extract_parameters()
        self.dfs = {}

    def _extract_parameters(self):
        """Extract parameters from filename"""
        match = re.search(r'(\d+)_([\d.]+)_(\d+)', self.simulation_file)
        if not match:
            raise ValueError("Filename pattern mismatch")
        return list(map(float, match.groups()))

    def _load_data(self):
        """Load and merge datasets"""
        self.dfs['sim'] = pd.read_csv(self.simulation_file).round(6)
        self.dfs['all'] = pd.read_csv(self.all_nodes_file).round(6)

        # Mark fluid nodes by matching X and Y coordinates
        merged = pd.merge(
            self.dfs['all'][['X (m)', 'Y (m)', 'Z (m)']],
            self.dfs['sim'][['X (m)', 'Y (m)']],
            on=['X (m)', 'Y (m)'],
            how='left',
            indicator=True
        )
        self.dfs['all']['is_fluid'] = (merged['_merge'] == 'both').astype(int)
        return self

    def _interpolate_fields(self, method='cubic'):
        """Interpolate missing values"""
        fluid_nodes = self.dfs['sim'][['X (m)', 'Y (m)'] + self.fields]
        solid_mask = self.dfs['all']['is_fluid'] == 0

        for field in self.fields:
            self.dfs['all'][field] = griddata(
                fluid_nodes[['X (m)', 'Y (m)']].values,
                fluid_nodes[field].values,
                self.dfs['all'][['X (m)', 'Y (m)']].values,
                method=method,
                fill_value=0
            )
        return self

    def _add_parameters(self):
        """Add simulation parameters to all nodes"""
        for i, param in enumerate(self.params):
            self.dfs['all'][f'param_{i}'] = param
        return self

    def plot_results(self):
        """Plot results"""
        fig, ax = plt.subplots(1, 2, figsize=(20, 7))

        # Compute velocity magnitude
        v_all = self.dfs['all'][['Velocity[i] (m/s)', 'Velocity[j] (m/s)', 'Velocity[k] (m/s)']].values
        velocity_magnitude = np.linalg.norm(v_all, axis=1)

        for i, field in enumerate(['Pressure (Pa)', 'Velocity Magnitude (m/s)']):
            fluid_mask = self.dfs['all']['is_fluid'] == 1
            solid_mask = self.dfs['all']['is_fluid'] == 0

            if field == 'Velocity Magnitude (m/s)':
                fluid_values = velocity_magnitude[fluid_mask]
                solid_values = velocity_magnitude[solid_mask]
            else:
                fluid_values = self.dfs['all'].loc[fluid_mask, field]
                solid_values = self.dfs['all'].loc[solid_mask, field]

            ax[i].scatter(
                self.dfs['all'].loc[solid_mask, 'X (m)'],
                self.dfs['all'].loc[solid_mask, 'Y (m)'],
                c=solid_values, s=20, cmap='viridis', label='Solid Nodes'
            )
            sc = ax[i].scatter(
                self.dfs['all'].loc[fluid_mask, 'X (m)'],
                self.dfs['all'].loc[fluid_mask, 'Y (m)'],
                c=fluid_values, s=20, cmap='viridis', marker='s', label='Fluid Nodes'
            )
            plt.colorbar(sc, ax=ax[i], label=field)
            ax[i].set_title(f'{field} Distribution')
            ax[i].axis('equal')
            ax[i].legend()

        plt.tight_layout()
        plt.show()

    def process(self, output_file=None, method='cubic', plot=False):
        """Full processing pipeline"""
        (self._load_data()
            ._interpolate_fields(method)
            ._add_parameters())
        if output_file is not None:
            self.dfs['all'].to_csv(output_file, index=False)
            print(f"Saved complete grid to {output_file}")
        if plot:
            self.plot_results()
        return self.dfs['all']


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    processor = FluidGridInterpolator(
        simulation_file='simfiles/45_24_0.0050.csv',
        all_nodes_file='all_nodes.csv'
    )
    processor.process(output_file='complete_simulation_grid.csv', method='cubic', plot=True)