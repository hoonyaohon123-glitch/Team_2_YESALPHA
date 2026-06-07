"""
Some of the funcitons that are used in the Jupyter Notebook:
    df_raw = load_data()
    check_quality(df_raw)
    df = clean_data(df_raw)
    univariate_analysis(df)
    bivariate_analysis(df)
    correlation_analysis(df)
    session_analysis(df)
    summary(df)
"""

import warnings
warnings.filterwarnings('ignore')

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── Global aesthetics ─────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 110

PALETTE = {
    'Low Activity':      '#4C72B0',
    'Moderate Activity': '#DD8452',
    'High Activity':     '#55A868',
}
ACTIVITY_ORDER = ['Low Activity', 'Moderate Activity', 'High Activity']
NUMERIC_COLS = [
    'Temperature', 'Humidity', 'CO2_InfraredSensor', 'CO2_ElectroChemicalSensor',
    'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2', 'MetalOxideSensor_Unit3',
    'MetalOxideSensor_Unit4', 'CO_GasSensor',
]
DB_PATH = 'ProjectData/gas_monitoring.db'


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Load the gas_monitoring table from the SQLite database.

    Returns
    -------
    df_raw : pd.DataFrame
        Raw, unmodified dataset.
    """
    conn = sqlite3.connect(db_path)
    df_raw = pd.read_sql('SELECT * FROM gas_monitoring', conn)
    conn.close()

    print(f'Shape        : {df_raw.shape}')
    print(f'Columns      : {list(df_raw.columns)}')
    print(f'Memory usage : {df_raw.memory_usage(deep=True).sum() / 1e6:.2f} MB')
    return df_raw


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Data quality assessment
# ─────────────────────────────────────────────────────────────────────────────
def check_quality(df: pd.DataFrame) -> None:
    """
    Print and visualise data quality issues:
    missing values, duplicates, and label noise in categorical columns.
    """
    # ── Missing values ──
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    print('=== Missing Values ===')
    print(missing_df.to_string())
    print(f'\n=== Duplicate Rows ===')
    print(f'  {df.duplicated().sum()} duplicate rows found')

    # ── Plot missing values ──
    if not missing_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        missing_df['Missing %'].sort_values().plot(kind='barh', ax=ax, color='#c0392b')
        ax.set_xlabel('Missing (%)')
        ax.set_title('Missing Value Rate per Column')
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.1f}%', (p.get_width() + 0.3, p.get_y() + 0.3))
        plt.tight_layout()
        plt.show()


    # ── Label noise ──
    print('\n=== Raw Activity Level values ===')
    print(df['Activity Level'].value_counts().to_string())
    print('\n=== Raw HVAC Operation Mode values ===')
    print(df['HVAC Operation Mode'].value_counts().to_string())

    # ── Physical boundary checks ──
    print('\n=== Physical Boundary Violations ===')
    print(f'  CO2_InfraredSensor negative values : {(df["CO2_InfraredSensor"] < 0).sum()} rows (min={df["CO2_InfraredSensor"].min():.2f} ppm)')
    print(f'  CO2_InfraredSensor mean            : {df["CO2_InfraredSensor"].mean():.2f} ppm (expected ~420+ ppm)')
    print(f'  CO2_ElectroChemicalSensor mean     : {df["CO2_ElectroChemicalSensor"].mean():.2f} ppm')
    print(f'  Humidity below 0%                  : {(df["Humidity"] < 0).sum()} rows')
    print(f'  Humidity above 100%                : {(df["Humidity"] > 100).sum()} rows')
    print(f'  Humidity range                     : {df["Humidity"].min():.2f}% to {df["Humidity"].max():.2f}%')


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Data cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the raw dataset:
    - Normalise Activity Level and HVAC Operation Mode labels
    - Replace temperature outliers (outside 10–40 C) with column median
    - Impute missing numeric values with median
    - Fix negative values in Humidity and CO_GasSensor (sensor faults)
    - Fill missing Ambient Light Level with mode

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataset.
    """
    df = df_raw.copy()

    # 1. Standardise Activity Level labels
    activity_map = {
        'Low Activity':      'Low Activity',
        'Low_Activity':      'Low Activity',
        'LowActivity':       'Low Activity',
        'Moderate Activity': 'Moderate Activity',
        'ModerateActivity':  'Moderate Activity',
        'High Activity':     'High Activity',
    }
    df['Activity Level'] = df['Activity Level'].map(activity_map)

    # 2. Standardise HVAC labels → lowercase + underscore
    df['HVAC Operation Mode'] = (
        df['HVAC Operation Mode']
        .str.strip()
        .str.lower()
        .str.replace(r'[\s\-]+', '_', regex=True)
    )

    # 3. Replace temperature outliers with median of valid readings
    TEMP_LOW, TEMP_HIGH = 10.0, 40.0
    temp_mask = (df['Temperature'] < TEMP_LOW) | (df['Temperature'] > TEMP_HIGH)
    temp_median = df.loc[~temp_mask, 'Temperature'].median()
    print(f'Temperature outliers : {temp_mask.sum()} rows  '
          f'(range before: {df_raw["Temperature"].min():.1f} – {df_raw["Temperature"].max():.1f} C)')
    df.loc[temp_mask, 'Temperature'] = temp_median
    print(f'  Replaced with median: {temp_median:.2f} C')

    # 4. Impute missing numeric columns with median
    for col in ['Humidity', 'MetalOxideSensor_Unit2', 'CO_GasSensor']:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f'  Imputed {col} with median = {median_val:.2f}')

    # 5a. Fix negative sensor values — physically impossible for Humidity and CO2 sensors.
    #     Humidity must be 0–100 (%RH); CO2 sensors cannot read below 0 ppm.
    #     Negative readings are likely sensor faults; replace with column median.
    for col in ['Humidity', 'CO_GasSensor', 'CO2_InfraredSensor']:
        neg_mask = df[col] < 0
        if neg_mask.sum() > 0:
            valid_median = df.loc[df[col] >= 0, col].median()
            print(f'  Negative values in {col}: {neg_mask.sum()} rows — replaced with median = {valid_median:.2f}')
            df.loc[neg_mask, col] = valid_median
        else:
            print(f'  No negative values found in {col}')

    # 5. Fill Ambient Light Level with mode
    light_mode = df['Ambient Light Level'].mode()[0]
    df['Ambient Light Level'].fillna(light_mode, inplace=True)
    print(f'  Imputed Ambient Light Level with mode = "{light_mode}"')

    print(f'\nRemaining nulls : {df.isnull().sum().sum()}')
    print('\nCleaned Activity Level distribution:')
    print(df['Activity Level'].value_counts().to_string())

    # 6. Remove duplicates if any
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f'\nRemoved {dup_count} duplicate rows.')
    
    # 7. Handling Null Values
    numerical_features = ['Temperature', 'Humidity', 'CO2_InfraredSensor', 'CO2_ElectroChemicalSensor', 'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2', 'MetalOxideSensor_Unit3', 'MetalOxideSensor_Unit4', 'CO_GasSensor'] # session id not included as it is not a feature for modeling
    categorical_features = ['Time of Day', 'HVAC Operation Mode', 'Ambient Light Level', 'Activity Level']
    print("\n -> Imputing missing values...")

    for col in ['Humidity', 'MetalOxideSensor_Unit2', 'CO_GasSensor']:
        if col in numerical_features:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        if 'Ambient Light Level' in categorical_features:
            df['Ambient Light Level'] = df.groupby('Time of Day')['Ambient Light Level'].transform(
                lambda x: x.fillna(x.mode()[0])
            )
    print("\nMissing values after imputation:")
    print(df[numerical_features + categorical_features].isnull().sum())


    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3b — Sanitize unphysical sensor contamination
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_contamination(df):
    """
    Explicitly strips out unphysical synthetic contamination
    before features are engineered, leaving true environmental outliers intact.

    Fixes:
    - Humidity outside 0-100% (physically impossible)
    - CO2_InfraredSensor negative values (gas concentration cannot be negative)
    - Repairs neutralized values using grouped medians by Time of Day
      to preserve natural daytime/nighttime environmental shifts
    """
    df = df.copy()

    # 1. Neutralize unphysical Humidity boundaries (Cannot be < 0% or > 100%)
    df.loc[(df['Humidity'] < 0) | (df['Humidity'] > 100), 'Humidity'] = np.nan

    # 2. Neutralize unphysical CO2 Infrared boundaries (Cannot have negative gas)
    df.loc[df['CO2_InfraredSensor'] < 0, 'CO2_InfraredSensor'] = np.nan

    # 3. Repair the neutralized values using grouped medians
    # We group by 'Time of Day' to preserve natural daytime/nighttime environmental shifts
    cols_to_repair = ['Humidity', 'CO2_InfraredSensor', 'MetalOxideSensor_Unit2', 'CO_GasSensor']
    for col in cols_to_repair:
        df[col] = df.groupby('Time of Day')[col].transform(lambda x: x.fillna(x.median()))

    print('=== Sanitization Complete ===')
    print(f'  Humidity range after fix        : {df["Humidity"].min():.2f}% to {df["Humidity"].max():.2f}%')
    print(f'  CO2_InfraredSensor min after fix: {df["CO2_InfraredSensor"].min():.2f} ppm')
    print(f'  Remaining nulls                 : {df.isnull().sum().sum()}')

    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Univariate analysis
# ─────────────────────────────────────────────────────────────────────────────
def univariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print summary statistics and plot distributions for all features.

    Returns
    -------
    desc : pd.DataFrame
        Descriptive statistics table with skewness and kurtosis.
    """
    # Summary stats
    desc = df[NUMERIC_COLS].describe().T
    desc['skewness'] = df[NUMERIC_COLS].skew().round(3)
    desc['kurtosis'] = df[NUMERIC_COLS].kurt().round(3)
    print(desc.round(3).to_string())

    # Skewness/Kurtosis interpretation
    # Skew > 0.5 = right-skewed (use log1p transform for linear models)
    # Skew < -0.5 = left-skewed
    # Kurt > 1 = leptokurtic = more extreme outliers than normal
    print('\n=== Skewness & Kurtosis Notes ===')
    for col in NUMERIC_COLS:
        skew = df[col].skew()
        kurt = df[col].kurt()
        if abs(skew) >= 0.5:
            print(f'  {col}: skew={skew:.2f} — consider log1p transform for linear models')
        if kurt > 1:
            print(f'  {col}: kurt={kurt:.2f} — heavy tails, more outliers than normal distribution')

    # Numeric distributions
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i]
        sns.histplot(df[col], bins=50, kde=True, ax=ax, color='#4C72B0', alpha=0.7)
        ax.set_title(f'C{i+1} - {col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        skew_val = df[col].skew()
        kurt_val = df[col].kurt()
        colour = '#c0392b' if abs(skew_val) >= 0.5 else '#27ae60'
        ax.annotate(f'skew={skew_val:.2f}  kurt={kurt_val:.2f}',
                    xy=(0.97, 0.92), xycoords='axes fraction',
                    ha='right', fontsize=8, color=colour)
    plt.suptitle('Distribution of Numeric Sensor Features (post-cleaning)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    # Categorical distributions
    cat_cols = ['Activity Level', 'Time of Day', 'HVAC Operation Mode', 'Ambient Light Level']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, col in enumerate(cat_cols):
        ax = axes[i]
        vc = df[col].value_counts()
        vc.plot(kind='bar', ax=ax, color=sns.color_palette('muted', len(vc)), edgecolor='white')
        ax.set_title(f'C{i+10} - {col}', fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2, p.get_height() + 10),
                        ha='center', fontsize=8)
    plt.suptitle('Categorical Feature Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return desc.round(3)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Bivariate analysis
# ─────────────────────────────────────────────────────────────────────────────
def bivariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Visualise how each feature relates to Activity Level via:
    - Box plots (numeric features)
    - Violin plots (CO2 sensors)
    - Stacked bar charts (categorical features)

    Returns
    -------
    group_means : pd.DataFrame
        Mean sensor readings grouped by activity level.
    """
    # Box plots — all numeric sensors
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        sns.boxplot(
            data=df, x='Activity Level', y=col, order=ACTIVITY_ORDER,
            palette=PALETTE, ax=axes[i], width=0.5,
            flierprops=dict(marker='o', markersize=2, alpha=0.3),
        )
        axes[i].set_title(col, fontweight='bold', fontsize=10)
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=15)
    plt.suptitle('Sensor Readings by Activity Level', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    # Violin plots — CO2 sensors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, col in enumerate(['CO2_InfraredSensor', 'CO2_ElectroChemicalSensor']):
        sns.violinplot(
            data=df, x='Activity Level', y=col, order=ACTIVITY_ORDER,
            palette=PALETTE, ax=axes[i], inner='quartile',
        )
        axes[i].set_title(f'{col} by Activity Level', fontweight='bold')
        axes[i].set_xlabel('')
    plt.suptitle('CO2 Sensor Distributions by Activity Level', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Stacked bar — categorical features
    cat_features = ['Time of Day', 'HVAC Operation Mode', 'Ambient Light Level']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, col in enumerate(cat_features):
        ct = pd.crosstab(df[col], df['Activity Level'], normalize='index') * 100
        ct = ct[[c for c in ACTIVITY_ORDER if c in ct.columns]]
        ct.plot(kind='bar', stacked=True, ax=axes[i],
                color=[PALETTE[c] for c in ct.columns], edgecolor='white', width=0.7)
        axes[i].set_title(f'{col} vs Activity Level', fontweight='bold')
        axes[i].set_ylabel('% of rows')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=35)
        axes[i].legend(loc='upper right', fontsize=8)
    plt.suptitle('Categorical Features vs Activity Level (normalised)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Group means table
    group_means = df.groupby('Activity Level')[NUMERIC_COLS].mean().round(2)
    print('\nGroup means by Activity Level:')
    print(group_means.loc[[a for a in ACTIVITY_ORDER if a in group_means.index]].to_string())

    return group_means.loc[[a for a in ACTIVITY_ORDER if a in group_means.index]]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────
def correlation_analysis(df: pd.DataFrame) -> pd.Series:
    """
    Compute and visualise Pearson correlations among numeric features
    and between features and the ordinally-encoded target.

    Returns
    -------
    target_corr : pd.Series
        Absolute correlations with Activity Level, sorted descending.
    """
    activity_encode = {'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}
    df = df.copy()
    df['Activity_Encoded'] = df['Activity Level'].map(activity_encode)

    corr_cols = NUMERIC_COLS + ['Activity_Encoded']
    corr_matrix = df[corr_cols].corr()

    # Full heatmap
    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, annot_kws={'size': 9},
    )
    ax.set_title('Pearson Correlation Matrix (Numeric Features + Encoded Target)',
                 fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()

    # Feature-target correlations ranked
    target_corr = corr_matrix['Activity_Encoded'].drop('Activity_Encoded').abs().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    target_corr.plot(kind='bar', ax=ax, color=sns.color_palette('Blues_r', len(target_corr)))
    ax.set_title('Feature Correlation with Activity Level (absolute Pearson r)', fontweight='bold')
    ax.set_ylabel('|Pearson r|')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=35)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}',
                    (p.get_x() + p.get_width() / 2, p.get_height() + 0.002),
                    ha='center', fontsize=9)
    plt.tight_layout()
    plt.show()

    print('\nFeature correlations with target (ranked):')
    print(target_corr.to_string())

    return target_corr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Session-level analysis
# ─────────────────────────────────────────────────────────────────────────────
def session_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse how sensor readings and activity distributions vary across sessions.

    Returns
    -------
    session_stats : pd.DataFrame
        Per-session aggregated statistics.
    """
    activity_encode = {'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}
    df = df.copy()
    df['Activity_Encoded'] = df['Activity Level'].map(activity_encode)

    session_stats = df.groupby('Session ID').agg(
        rows=('Activity Level', 'count'),
        low_pct=('Activity_Encoded', lambda x: (x == 0).mean() * 100),
        mod_pct=('Activity_Encoded', lambda x: (x == 1).mean() * 100),
        high_pct=('Activity_Encoded', lambda x: (x == 2).mean() * 100),
        avg_co2=('CO2_ElectroChemicalSensor', 'mean'),
        avg_temp=('Temperature', 'mean'),
    ).reset_index()

    print(f'Unique sessions : {df["Session ID"].nunique()}')
    print(f'Rows per session — min: {session_stats["rows"].min()}, '
          f'max: {session_stats["rows"].max()}, '
          f'mean: {session_stats["rows"].mean():.1f}')

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    session_stats['rows'].plot(kind='hist', bins=30, ax=axes[0], color='#4C72B0', edgecolor='white')
    axes[0].set_title('Distribution of Rows per Session', fontweight='bold')
    axes[0].set_xlabel('Number of rows')

    axes[1].scatter(session_stats['Session ID'], session_stats['avg_co2'],
                    alpha=0.5, s=15, color='#DD8452')
    axes[1].set_title('Mean CO2 (ElectroChemical) per Session', fontweight='bold')
    axes[1].set_xlabel('Session ID')
    axes[1].set_ylabel('Mean CO2 (ppm)')
    plt.tight_layout()
    plt.show()

    # Activity split across top 20 sessions
    sample_sessions = df['Session ID'].value_counts().head(20).index
    act_by_session = (
        df[df['Session ID'].isin(sample_sessions)]
        .groupby(['Session ID', 'Activity Level'])
        .size()
        .unstack(fill_value=0)
    )
    act_pct = act_by_session.div(act_by_session.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(14, 5))
    act_pct[[c for c in ACTIVITY_ORDER if c in act_pct.columns]].plot(
        kind='bar', stacked=True, ax=ax,
        color=[PALETTE[c] for c in ACTIVITY_ORDER if c in act_pct.columns],
        edgecolor='white',
    )
    ax.set_title('Activity Level Distribution across Sessions (top 20)', fontweight='bold')
    ax.set_xlabel('Session ID')
    ax.set_ylabel('% of readings')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return session_stats


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Summary
# ─────────────────────────────────────────────────────────────────────────────
def summary(df: pd.DataFrame, save_csv: bool = True) -> None:
    """
    Print the final cleaned dataset summary and optionally save cleaned CSV.

    Parameters
    ----------
    df       : cleaned DataFrame (output of clean_data)
    save_csv : if True, saves data/cleaned_gas_monitoring.csv # NOTE!!
    """
    print('=== Final Cleaned Dataset ===')
    print(f'Shape         : {df.shape}')
    print(f'Missing values: {df.isnull().sum().sum()}')
    print()
    print('Activity Level distribution:')
    vc = df['Activity Level'].value_counts()
    for k, v in vc.items():
        print(f'  {k:<22}: {v:>5}  ({v / len(df) * 100:.1f}%)')

    if save_csv:
        import os
        os.makedirs('data', exist_ok=True)
        out_path = 'data/cleaned_gas_monitoring.csv'
        df.to_csv(out_path, index=False)
        print(f'\nCleaned dataset saved to {out_path}')

    # Modelling pipeline reminder
    print('\n=== Next Step: Modelling Pipeline ===')
    print('1. Encode features    : ordinal (Time of Day, Ambient Light), one-hot (HVAC)')
    print('2. Transform skewed   : log1p on CO_GasSensor for linear models')
    print('3. Split data         : GroupShuffleSplit by Session ID (80/20)')
    print('4. Handle imbalance   : class_weight="balanced" OR SMOTE on train set only')
    print('5. Cross-validate     : StratifiedGroupKFold (5 folds)')
    print('6. Train models       : Logistic Regression → Random Forest → XGBoost')
    print('7. Evaluate           : macro F1-score (not accuracy — classes are imbalanced)')
