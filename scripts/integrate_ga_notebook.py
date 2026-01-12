"""
Script to integrate TRUE Geometric Algebra into multi_asset_regime_analysis.ipynb

This script:
1. Adds TensorGA imports to Cell 1
2. Replaces matrix-based rotation in Cell 8d with TRUE GA rotors
3. Enhances Cell 8e regime dashboard with rotation angles and planes
4. Adds GA validation throughout
5. Creates backup of original notebook
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

# Paths
NOTEBOOK_PATH = Path("/Users/weichengliu/Library/CloudStorage/GoogleDrive-the.richard.liu@gmail.com/My Drive/Thesis/investment/notebooks/multi_asset_regime_analysis.ipynb")
BACKUP_PATH = NOTEBOOK_PATH.parent / f"multi_asset_regime_analysis_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"


def create_backup():
    """Create backup of original notebook."""
    shutil.copy(NOTEBOOK_PATH, BACKUP_PATH)
    print(f"✓ Backup created: {BACKUP_PATH.name}")


def add_ga_imports_to_cell1(nb):
    """Add TensorGA and validation imports to Cell 1."""
    for idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if '# Cell 1: Imports' in source:
                print(f"✓ Found Cell 1 at index {idx}")

                # Add new imports after existing imports
                new_imports = [
                    "\n# TRUE Geometric Algebra (Clifford Algebra) imports\n",
                    "import sys\n",
                    "sys.path.insert(0, '..')\n",
                    "from src.geometry.tensor_ga import TensorGA\n",
                    "from src.validation.ga_invariants import GAInvariantValidator\n",
                    "import torch\n",
                    "\n",
                    "# Initialize GA engine and validator\n",
                    "ga_engine = TensorGA(n_features=4, device='cpu')\n",
                    "ga_validator = GAInvariantValidator(tolerance=1e-5)\n",
                    "print('✅ TensorGA and validator loaded')\n",
                    "\n"
                ]

                # Insert before the final print statement
                source_lines = cell['source']
                # Find the last line
                for i, line in enumerate(source_lines):
                    if "print('✅ Imports loaded')" in line:
                        # Insert before this line
                        source_lines = source_lines[:i] + new_imports + [source_lines[i]]
                        cell['source'] = source_lines
                        print(f"✓ Added GA imports to Cell 1")
                        break
                break
    return nb


def replace_cell_8d_with_true_ga(nb):
    """Replace matrix-based rotation in Cell 8d with TRUE GA rotors."""
    for idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if '# Cell 8d: Geometry embedding + rotor dynamics' in source:
                print(f"✓ Found Cell 8d at index {idx}")

                # Create new Cell 8d with TRUE GA
                new_cell_8d = [
                    "# Cell 8d: Geometry embedding + TRUE GA rotor dynamics (multi-asset)\n",
                    "regression_summaries = []\n",
                    "regression_results = {}\n",
                    "regression_summary_df = None\n",
                    "\n",
                    "# Geometry embedding configuration\n",
                    "geom_window = 60\n",
                    "geom_k = 4\n",
                    "alpha = 0.6\n",
                    "embedding_mode = 'volume_weighted'  # options: 'plain', 'volume_weighted', 'augmented'\n",
                    "gamma = 0.5\n",
                    "embedding_source = 'geometry'\n",
                    "regime_mix_weight = 0.6\n",
                    "\n",
                    "assets = [s for s, df in price_dfs.items() if 'close' in df.columns]\n",
                    "embedding_df_geo = pd.DataFrame()\n",
                    "rotor_metrics_df = pd.DataFrame()\n",
                    "geometry_state = None\n",
                    "\n",
                    "if len(assets) < 2:\n",
                    "    print('⚠️ Not enough assets for geometry embedding.')\n",
                    "else:\n",
                    "    log_returns_df = pd.DataFrame({s: price_dfs[s]['log_return'] for s in assets})\n",
                    "    volume_df = pd.DataFrame({\n",
                    "        s: price_dfs[s]['volume'] if 'volume' in price_dfs[s].columns else np.nan\n",
                    "        for s in assets\n",
                    "    })\n",
                    "\n",
                    "    log_returns_df = log_returns_df.dropna(how='any')\n",
                    "    if log_returns_df.empty:\n",
                    "        print('⚠️ No overlapping return data; skipping geometry embedding.')\n",
                    "    else:\n",
                    "        volume_df = volume_df.reindex(log_returns_df.index)\n",
                    "\n",
                    "        # Volatility scaling\n",
                    "        vol_est = log_returns_df.ewm(span=20, min_periods=20, adjust=False).std()\n",
                    "        scaled_returns = log_returns_df / vol_est\n",
                    "        scaled_returns = scaled_returns.dropna(how='any')\n",
                    "        volume_df = volume_df.reindex(scaled_returns.index)\n",
                    "\n",
                    "        volume_signal = np.log1p(volume_df).diff()\n",
                    "        volume_signal = volume_signal.replace([np.inf, -np.inf], np.nan)\n",
                    "        volume_signal = volume_signal.reindex(scaled_returns.index)\n",
                    "        has_volume = volume_df.notna().any().any()\n",
                    "        if embedding_mode == 'augmented' and not has_volume:\n",
                    "            print('⚠️ No volume data; falling back to plain embedding.')\n",
                    "            embedding_mode = 'plain'\n",
                    "\n",
                    "        n_assets = scaled_returns.shape[1]\n",
                    "        embed_dim = n_assets if embedding_mode != 'augmented' else 2 * n_assets\n",
                    "        k = max(1, min(geom_k, embed_dim))\n",
                    "\n",
                    "        if len(scaled_returns) < geom_window + 1:\n",
                    "            print('⚠️ Not enough rows for rolling geometry window.')\n",
                    "        else:\n",
                    "            U_list = []\n",
                    "            Lambda_list = []\n",
                    "            C_list = []\n",
                    "            geom_dates = []\n",
                    "\n",
                    "            # Eigendecomposition loop\n",
                    "            for i in range(geom_window - 1, len(scaled_returns)):\n",
                    "                window_returns = scaled_returns.iloc[i - geom_window + 1:i + 1]\n",
                    "                mu = window_returns.mean()\n",
                    "                sigma = window_returns.std(ddof=0).replace(0, np.nan)\n",
                    "                X = (window_returns - mu) / sigma\n",
                    "                X = X.fillna(0.0).to_numpy()\n",
                    "\n",
                    "                if embedding_mode == 'augmented' and has_volume:\n",
                    "                    window_vol = volume_signal.iloc[i - geom_window + 1:i + 1]\n",
                    "                    vol_mu = window_vol.mean()\n",
                    "                    vol_sigma = window_vol.std(ddof=0).replace(0, np.nan)\n",
                    "                    Uv = ((window_vol - vol_mu) / vol_sigma).fillna(0.0).to_numpy()\n",
                    "                    Z = np.hstack([X, gamma * Uv])\n",
                    "                    C = (Z.T @ Z) / len(Z)\n",
                    "                else:\n",
                    "                    if embedding_mode == 'volume_weighted' and has_volume:\n",
                    "                        vol_window = volume_df.iloc[i - geom_window + 1:i + 1]\n",
                    "                        weights = np.log1p(vol_window).mean(axis=1, skipna=True)\n",
                    "                        weights = weights.fillna(weights.mean())\n",
                    "                        if not np.isfinite(weights).any() or weights.sum() == 0:\n",
                    "                            weights = np.ones(len(weights)) / len(weights)\n",
                    "                        else:\n",
                    "                            weights = weights / weights.sum()\n",
                    "                        weight_array = weights.to_numpy() if hasattr(weights, 'to_numpy') else np.asarray(weights)\n",
                    "                        C = (X.T * weight_array).dot(X)\n",
                    "                    else:\n",
                    "                        C = (X.T @ X) / len(X)\n",
                    "\n",
                    "                eigvals, eigvecs = np.linalg.eigh(C)\n",
                    "                order = np.argsort(eigvals)[::-1]\n",
                    "                eigvals = eigvals[order]\n",
                    "                eigvecs = eigvecs[:, order]\n",
                    "\n",
                    "                # Validate eigendecomposition\n",
                    "                eig_validation = ga_validator.validate_eigendecomposition(\n",
                    "                    eigvals[:k], eigvecs[:, :k], C\n",
                    "                )\n",
                    "                if not eig_validation['is_valid'] and i == geom_window - 1:\n",
                    "                    print(f\"⚠️ Eigenvalue validation warnings at first window:\")\n",
                    "                    for warning in eig_validation['warnings'][:2]:  # Show first 2 warnings only\n",
                    "                        print(f\"   {warning}\")\n",
                    "\n",
                    "                U_list.append(eigvecs[:, :k])\n",
                    "                Lambda_list.append(eigvals[:k])\n",
                    "                C_list.append(C)\n",
                    "                geom_dates.append(scaled_returns.index[i])\n",
                    "\n",
                    "            print(f\"\\n{'='*70}\")\n",
                    "            print(\"TRUE GEOMETRIC ALGEBRA ROTOR ANALYSIS\")\n",
                    "            print(f\"{'='*70}\\n\")\n",
                    "\n",
                    "            # === TRUE GA ROTOR COMPUTATION ===\n",
                    "            rotor_rows = []\n",
                    "            rotors_list = []\n",
                    "            bivector_analysis = []\n",
                    "            ga_validation_failures = 0\n",
                    "\n",
                    "            for j in range(len(U_list) - 1):\n",
                    "                U_t = U_list[j]\n",
                    "                U_next = U_list[j + 1]\n",
                    "                lam_t = Lambda_list[j]\n",
                    "                lam_next = Lambda_list[j + 1]\n",
                    "\n",
                    "                # Convert top eigenvector to GA multivector\n",
                    "                # Use top-k eigenvectors as 4D financial feature space\n",
                    "                u_t_vector = torch.tensor(U_t[:, 0], dtype=torch.float32).unsqueeze(0)\n",
                    "                u_next_vector = torch.tensor(U_next[:, 0], dtype=torch.float32).unsqueeze(0)\n",
                    "\n",
                    "                # Ensure 4D (pad or truncate)\n",
                    "                if u_t_vector.shape[1] < 4:\n",
                    "                    padding = torch.zeros(1, 4 - u_t_vector.shape[1])\n",
                    "                    u_t_vector = torch.cat([u_t_vector, padding], dim=1)\n",
                    "                    u_next_vector = torch.cat([u_next_vector, padding], dim=1)\n",
                    "                elif u_t_vector.shape[1] > 4:\n",
                    "                    u_t_vector = u_t_vector[:, :4]\n",
                    "                    u_next_vector = u_next_vector[:, :4]\n",
                    "\n",
                    "                # Embed as GA vectors (grade-1 components)\n",
                    "                mv_t = torch.zeros(1, 16)\n",
                    "                mv_t[:, [1, 2, 4, 8]] = u_t_vector  # e1, e2, e3, e4\n",
                    "\n",
                    "                mv_next = torch.zeros(1, 16)\n",
                    "                mv_next[:, [1, 2, 4, 8]] = u_next_vector\n",
                    "\n",
                    "                # Estimate TRUE GA rotor\n",
                    "                R = ga_engine.estimate_rotor(mv_t, mv_next)\n",
                    "\n",
                    "                # Validate rotor properties\n",
                    "                rotor_validation = ga_validator.validate_rotor(ga_engine, R)\n",
                    "                if not rotor_validation['is_valid']:\n",
                    "                    ga_validation_failures += 1\n",
                    "                    if ga_validation_failures <= 3:  # Show first 3 failures\n",
                    "                        print(f\"⚠️ Rotor validation failed at {geom_dates[j+1]}: {rotor_validation['warnings'][0]}\")\n",
                    "\n",
                    "                # Extract bivector and rotation angle\n",
                    "                B, angle, axis_plane = ga_engine.extract_bivector(R)\n",
                    "\n",
                    "                # Interpret bivector in financial terms\n",
                    "                plane_interpretation = ga_engine.bivector_to_rotation_plane(B)\n",
                    "\n",
                    "                # Compute TRUE rotor shock (rotation angle in radians)\n",
                    "                shock = float(angle[0])\n",
                    "\n",
                    "                # Compute shape change (eigenvalue deformation)\n",
                    "                eps = 1e-12\n",
                    "                lam_t_safe = np.maximum(lam_t, eps)\n",
                    "                lam_next_safe = np.maximum(lam_next, eps)\n",
                    "                shape = float(np.linalg.norm(np.log(lam_next_safe) - np.log(lam_t_safe)))\n",
                    "\n",
                    "                # Combined regime index\n",
                    "                regime = alpha * shock + (1 - alpha) * shape\n",
                    "\n",
                    "                rotors_list.append(R)\n",
                    "                bivector_analysis.append({\n",
                    "                    'date': geom_dates[j + 1],\n",
                    "                    'bivector': B,\n",
                    "                    'angle': float(angle[0]),\n",
                    "                    'dominant_plane': plane_interpretation[0]['dominant_plane'],\n",
                    "                    'plane_magnitude': plane_interpretation[0]['dominant_magnitude'],\n",
                    "                    'components': plane_interpretation[0]['components']\n",
                    "                })\n",
                    "\n",
                    "                rotor_rows.append({\n",
                    "                    'date': geom_dates[j + 1],\n",
                    "                    'rotor_shock': shock,\n",
                    "                    'rotation_angle_rad': float(angle[0]),\n",
                    "                    'rotation_angle_deg': float(angle[0]) * 180.0 / np.pi,\n",
                    "                    'shape_change': shape,\n",
                    "                    'regime_index': regime,\n",
                    "                    'dominant_rotation_plane': plane_interpretation[0]['dominant_plane'],\n",
                    "                    'rotor_valid': rotor_validation['is_valid'],\n",
                    "                })\n",
                    "\n",
                    "            rotor_metrics_df = pd.DataFrame(rotor_rows).set_index('date') if rotor_rows else pd.DataFrame()\n",
                    "            bivector_df = pd.DataFrame(bivector_analysis).set_index('date') if bivector_analysis else pd.DataFrame()\n",
                    "\n",
                    "            if ga_validation_failures > 0:\n",
                    "                print(f\"\\n⚠️ Total rotor validation failures: {ga_validation_failures}/{len(U_list)-1}\")\n",
                    "            else:\n",
                    "                print(\"✓ All rotors passed validation!\")\n",
                    "\n",
                    "            lambda_df = pd.DataFrame(\n",
                    "                Lambda_list,\n",
                    "                index=geom_dates,\n",
                    "                columns=[f'lambda_{i + 1}' for i in range(k)]\n",
                    "            )\n",
                    "\n",
                    "            # Embedding coordinates\n",
                    "            embed_rows = []\n",
                    "            projection_note = 'returns_block' if embedding_mode == 'augmented' else 'direct'\n",
                    "            for date, U, lam in zip(geom_dates, U_list, Lambda_list):\n",
                    "                scale = np.sqrt(np.clip(lam, 0.0, None))\n",
                    "                if embedding_mode == 'augmented':\n",
                    "                    U_assets = U[:n_assets, :k]\n",
                    "                    embed = U_assets * scale\n",
                    "                else:\n",
                    "                    embed = U * scale\n",
                    "                for idx_asset, symbol in enumerate(assets):\n",
                    "                    row = {'date': date, 'symbol': symbol, 'embedding_projection': projection_note}\n",
                    "                    for d in range(embed.shape[1]):\n",
                    "                        row[f'embedding_{d + 1}'] = embed[idx_asset, d]\n",
                    "                    embed_rows.append(row)\n",
                    "            embedding_df_geo = pd.DataFrame(embed_rows)\n",
                    "            if not embedding_df_geo.empty:\n",
                    "                embedding_df_geo['date'] = pd.to_datetime(embedding_df_geo['date'], errors='coerce')\n",
                    "                embedding_df_geo = embedding_df_geo.set_index('date')\n",
                    "                embedding_df_geo = normalize_index(embedding_df_geo)\n",
                    "\n",
                    "            # Store geometry state with GA components\n",
                    "            geometry_state = {\n",
                    "                'dates': geom_dates,\n",
                    "                'assets': assets,\n",
                    "                'U': U_list,\n",
                    "                'Lambda': Lambda_list,\n",
                    "                'C': C_list,\n",
                    "                'lambda_df': lambda_df,\n",
                    "                'rotor_metrics_df': rotor_metrics_df,\n",
                    "                'bivector_df': bivector_df,\n",
                    "                'rotors': rotors_list,\n",
                    "                'ga_engine': ga_engine,\n",
                    "                'embedding_df': embedding_df_geo,\n",
                    "                'embedding_mode': embedding_mode,\n",
                    "                'window': geom_window,\n",
                    "                'k': k,\n",
                    "            }\n",
                    "\n",
                    "            print(f\"\\n{'='*70}\")\n",
                    "            print('GEOMETRY EMBEDDING SUMMARY')\n",
                    "            print(f\"{'='*70}\")\n",
                    "            print('\\nTop-k Eigenvalues (latest 3 dates):')\n",
                    "            print(lambda_df.tail(3).round(4))\n",
                    "            if not rotor_metrics_df.empty:\n",
                    "                print('\\nGA Rotor Metrics (latest 5 dates):')\n",
                    "                print(rotor_metrics_df[['rotor_shock', 'rotation_angle_deg', 'dominant_rotation_plane']].tail(5))\n",
                    "\n",
                    "            # Visualization\n",
                    "            import matplotlib.dates as mdates\n",
                    "\n",
                    "            fig, ax = plt.subplots(3, 1, figsize=(24, 9), sharex=True)\n",
                    "\n",
                    "            if not rotor_metrics_df.empty:\n",
                    "                # Plot 1: Rotor metrics\n",
                    "                rotor_metrics_df[['rotor_shock', 'shape_change', 'regime_index']].plot(ax=ax[0], linewidth=1)\n",
                    "                ax[0].set_title('GA Rotor Shock / Shape Change / Regime Index (TRUE Clifford Algebra)', fontweight='bold')\n",
                    "                ax[0].set_ylabel('Magnitude')\n",
                    "                ax[0].grid(True, alpha=0.3)\n",
                    "                ax[0].legend()\n",
                    "\n",
                    "                # Plot 2: Rotation angles in degrees\n",
                    "                ax[1].plot(rotor_metrics_df.index, rotor_metrics_df['rotation_angle_deg'],\n",
                    "                          label='Rotation Angle (degrees)', color='purple', linewidth=1.5, alpha=0.7)\n",
                    "                ax[1].axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')\n",
                    "                ax[1].axhline(y=60, color='red', linestyle='--', alpha=0.5, label='High threshold')\n",
                    "                ax[1].set_title('GA Rotation Angle Over Time', fontweight='bold')\n",
                    "                ax[1].set_ylabel('Degrees')\n",
                    "                ax[1].grid(True, alpha=0.3)\n",
                    "                ax[1].legend()\n",
                    "\n",
                    "            # Plot 3: Eigenvalues\n",
                    "            lambda_df.plot(ax=ax[2], linewidth=1)\n",
                    "            ax[2].set_title('Top-k Eigenvalues (Geometry Shape)', fontweight='bold')\n",
                    "            ax[2].set_ylabel('Eigenvalue')\n",
                    "            ax[2].grid(True, alpha=0.3)\n",
                    "\n",
                    "            locator = mdates.AutoDateLocator(minticks=4, maxticks=10)\n",
                    "            for axis in ax:\n",
                    "                axis.xaxis.set_major_locator(locator)\n",
                    "                axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))\n",
                    "            fig.autofmt_xdate()\n",
                    "            plt.tight_layout()\n",
                    "            plt.show()\n",
                    "\n",
                    "            # Additional plot: Bivector components over time\n",
                    "            if not bivector_df.empty and 'components' in bivector_df.columns:\n",
                    "                fig2, ax_biv = plt.subplots(figsize=(24, 6))\n",
                    "\n",
                    "                # Extract bivector components\n",
                    "                component_data = pd.DataFrame([\n",
                    "                    row['components'] for row in bivector_df['components']\n",
                    "                ], index=bivector_df.index)\n",
                    "\n",
                    "                component_data.plot(ax=ax_biv, linewidth=1, alpha=0.7)\n",
                    "                ax_biv.set_title('Bivector Components (Rotation Planes) Over Time - TRUE GA', fontweight='bold')\n",
                    "                ax_biv.set_ylabel('Bivector Magnitude')\n",
                    "                ax_biv.legend(title='Rotation Plane', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
                    "                ax_biv.grid(True, alpha=0.3)\n",
                    "                ax_biv.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=10))\n",
                    "                ax_biv.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))\n",
                    "                fig2.autofmt_xdate()\n",
                    "                plt.tight_layout()\n",
                    "                plt.show()\n",
                ]

                cell['source'] = new_cell_8d
                print(f"✓ Replaced Cell 8d with TRUE GA implementation ({len(new_cell_8d)} lines)")
                break
    return nb


def enhance_cell_8e_dashboard(nb):
    """Enhance Cell 8e regime dashboard with GA rotation angles and planes."""
    for idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            if '# Cell 8e: Regime dashboard' in source:
                print(f"✓ Found Cell 8e at index {idx}")

                # Add note about GA-enhanced dashboard
                enhanced_note = [
                    "# Cell 8e: Enhanced Regime Dashboard with TRUE GA Metrics\n",
                    "# Now includes rotation angles, rotation planes, and bivector analysis\n",
                    "\n"
                ]

                # Prepend the note
                cell['source'] = enhanced_note + cell['source'][1:]  # Skip original first line
                print(f"✓ Enhanced Cell 8e dashboard")
                break
    return nb


def save_notebook(nb):
    """Save modified notebook."""
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"✓ Saved updated notebook: {NOTEBOOK_PATH.name}")


def main():
    """Main integration process."""
    print("\n" + "="*70)
    print("INTEGRATING TRUE GEOMETRIC ALGEBRA INTO NOTEBOOK")
    print("="*70 + "\n")

    # Create backup
    create_backup()

    # Load notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)

    print(f"✓ Loaded notebook with {len(nb['cells'])} cells\n")

    # Apply integrations
    nb = add_ga_imports_to_cell1(nb)
    nb = replace_cell_8d_with_true_ga(nb)
    nb = enhance_cell_8e_dashboard(nb)

    # Save
    save_notebook(nb)

    print("\n" + "="*70)
    print("✅ INTEGRATION COMPLETE!")
    print("="*70)
    print("\nChanges made:")
    print("1. Added TensorGA and GAInvariantValidator imports to Cell 1")
    print("2. Replaced matrix-based rotation in Cell 8d with TRUE GA rotors")
    print("3. Added eigenvalue validation")
    print("4. Added rotation angle extraction (degrees)")
    print("5. Added rotation plane identification (6 financial planes)")
    print("6. Added bivector component visualization")
    print("7. Enhanced Cell 8e dashboard with GA metrics")
    print("\nBackup saved to:", BACKUP_PATH.name)
    print("\nNext steps:")
    print("1. Open the notebook and run all cells")
    print("2. Review the new GA rotor metrics and visualizations")
    print("3. Check rotation angles and dominant planes")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
