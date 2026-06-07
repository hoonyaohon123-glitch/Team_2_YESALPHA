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

# Colours for plotting graphs for activity classes
PALETTE = {
    'Low Activity':      '#4C72B0',
    'Moderate Activity': '#DD8452',
    'High Activity':     '#55A868',
}

# Ordering of the activity (Low -> Moderate -> High)
ACTIVITY_ORDER = ['Low Activity', 'Moderate Activity', 'High Activity']

# Listing all the numeric sensor columns that will be used for anlyzing
NUMERIC_COLS = [
    'Temperature', 'Humidity', 'CO2_InfraredSensor', 'CO2_ElectroChemicalSensor',
    'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2', 'MetalOxideSensor_Unit3',
    'MetalOxideSensor_Unit4', 'CO_GasSensor',
]

# Setting the default file path
DB_PATH = 'ProjectData/gas_monitoring.db'


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Load data
# ─────────────────────────────────────────────────────────────────────────────
# Function for loading the data and displaying basic info
def load_data(db_path: str = DB_PATH) -> pd.DataFrame:

    "Load the gas_monitoring table from the SQLite database."

    conn = sqlite3.connect(db_path)
    df_raw = pd.read_sql('SELECT * FROM gas_monitoring', conn)
    conn.close()

    print(f'Shape        : {df_raw.shape}')
    print(f'Columns      : {list(df_raw.columns)}')
    print(f'Memory usage : {df_raw.memory_usage(deep=True).sum() / 1e6:.2f} MB')
    return df_raw


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Data quality assessment
# ─────────────────────────────────────────────────────────────────────────────
# Function for data quality checking to be standardized
def check_quality(df: pd.DataFrame) -> None:

    "Print and visualise data quality issues: missing values, duplicates, and label noise in categorical columns."

    # Missing values - Conveerts, counts and displays the values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    missing_df = missing_df[missing_df['Missing Count'] > 0]

    print('=== Missing Values ===')
    print(missing_df.to_string())
    print(f'\n=== Duplicate Rows ===')
    print(f'  {df.duplicated().sum()} duplicate rows found')

    # ── Plot missing values ──
    # If statement: plots a chart to show missing values
    if not missing_df.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        missing_df['Missing %'].sort_values().plot(kind='barh', ax=ax, color='#c0392b')
        ax.set_xlabel('Missing (%)')
        ax.set_title('Missing Value Rate per Column')

        # For loop: label each bar with its exact missing % value 
        for p in ax.patches:
            ax.annotate(f'{p.get_width():.1f}%', (p.get_width() + 0.3, p.get_y() + 0.3))
        plt.tight_layout()
        plt.show()


    # Label noise - detect inconsitenct such as spacing, undescore, cases etc.
    print('\n=== Raw Activity Level values ===')
    print(df['Activity Level'].value_counts().to_string())
    print('\n=== Raw HVAC Operation Mode values ===')
    print(df['HVAC Operation Mode'].value_counts().to_string())

    # Physical boundary checks - Flags out physically impossible values
    print('\n=== Physical Boundary Violations ===')
    print(f'  CO2_InfraredSensor negative values : {(df["CO2_InfraredSensor"] < 0).sum()} rows (min={df["CO2_InfraredSensor"].min():.2f} ppm)')
    print(f'  CO2_InfraredSensor mean            : {df["CO2_InfraredSensor"].mean():.2f} ppm (expected ~420+ ppm)')
    print(f'  CO2_ElectroChemicalSensor mean     : {df["CO2_ElectroChemicalSensor"].mean():.2f} ppm')
    print(f'  Humidity below 0%                  : {(df["Humidity"] < 0).sum()} rows')
    print(f'  Humidity above 100%                : {(df["Humidity"] > 100).sum()} rows')
    print(f'  Humidity range                     : {df["Humidity"].min():.2f}% to {df["Humidity"].max():.2f}%')


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Data cleaning
# ─────────────────────────────────────────────────────────────────────────────
# Data cleaning based on the data quality assessment
def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:

    "Clean and standardise the raw dataset"

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
    # String operations: removes any spaces, lowercase and replace spaces with underscores
    df['HVAC Operation Mode'] = (
        df['HVAC Operation Mode']
        .str.strip()
        .str.lower()
        .str.replace(r'[\s\-]+', '_', regex=True)
    )

    # 3. Replace temperature outliers with median of valid readings
    TEMP_LOW, TEMP_HIGH = 10.0, 40.0

    # Boolean mask: identifies rows where temperature is outside valid range
    temp_mask = (df['Temperature'] < TEMP_LOW) | (df['Temperature'] > TEMP_HIGH)
    temp_median = df.loc[~temp_mask, 'Temperature'].median()
    print(f'Temperature outliers : {temp_mask.sum()} rows  '
          f'(range before: {df_raw["Temperature"].min():.1f} – {df_raw["Temperature"].max():.1f} C)')
    df.loc[temp_mask, 'Temperature'] = temp_median
    print(f'  Replaced with median: {temp_median:.2f} C')

    # 4. Impute missing numeric columns with median
    # For loop: fills in any missing values in each numeric column with the median
    for col in ['Humidity', 'MetalOxideSensor_Unit2', 'CO_GasSensor']:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f'  Imputed {col} with median = {median_val:.2f}')

    # 5a. Fix negative sensor values — physically impossible for Humidity and CO2 sensors.
    # For loop: checks each sensor column for negative values and replaces with median
    for col in ['Humidity', 'CO_GasSensor', 'CO2_InfraredSensor']:
        neg_mask = df[col] < 0

        # If-else statement: the values will only be replaced if negative values actually exist
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
    # Counts the amount of duplicates
    dup_count = df.duplicated().sum()

    # If statement: removes duplicates if there are any found
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f'\nRemoved {dup_count} duplicate rows.')
    
    # 7. Handling Null Values

    # List all numeric columns that may need imputation
    numerical_features = ['Temperature', 'Humidity', 'CO2_InfraredSensor', 'CO2_ElectroChemicalSensor', 'MetalOxideSensor_Unit1', 'MetalOxideSensor_Unit2', 'MetalOxideSensor_Unit3', 'MetalOxideSensor_Unit4', 'CO_GasSensor'] # session id not included as it is not a feature for modeling
    
    # List all categorical columns that may need imputation
    categorical_features = ['Time of Day', 'HVAC Operation Mode', 'Ambient Light Level', 'Activity Level']
    print("\n -> Imputing missing values...")

    # For loop: filling missing values for the columns using median (Numeric columns)
    for col in ['Humidity', 'MetalOxideSensor_Unit2', 'CO_GasSensor']:

        # If statement: imputes if its in the numeric features list
        if col in numerical_features:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        # If statement: fills in missing ambient light using common values 
        if 'Ambient Light Level' in categorical_features:
            df['Ambient Light Level'] = df.groupby('Time of Day')['Ambient Light Level'].transform(
                lambda x: x.fillna(x.mode()[0])
            )

    print("\nMissing values after imputation:")
    print(df[numerical_features + categorical_features].isnull().sum())


    return df

# ─────────────────────────────────────────────────────────────────────────────
# PART 3b — Sanitize unphysical sensor contamination
# ─────────────────────────────────────────────────────────────────────────────
# Function for checking for any unrealistic data types
def sanitize_contamination(df):

    "Strips out unphysical synthetic contamination before any of thefeatures are engineered."

    df = df.copy()

    # 1. Neutralize unphysical Humidity boundaries (Cannot be < 0% or > 100%)
    df.loc[(df['Humidity'] < 0) | (df['Humidity'] > 100), 'Humidity'] = np.nan

    # 2. Neutralize unphysical CO2 Infrared boundaries (Cannot have negative gas)
    df.loc[df['CO2_InfraredSensor'] < 0, 'CO2_InfraredSensor'] = np.nan

    # 3. Repair the neutralized values using grouped medians
    # We group by 'Time of Day' to preserve natural daytime/nighttime environmental shifts
    cols_to_repair = ['Humidity', 'CO2_InfraredSensor', 'MetalOxideSensor_Unit2', 'CO_GasSensor']

    # For loop: repairs any contaminated columns using the median which is grouped by Time of Day
    for col in cols_to_repair:
        df[col] = df.groupby('Time of Day')[col].transform(lambda x: x.fillna(x.median()))

    print('=== Sanitization Complete ===')
    print(f'  Humidity range after fix        : {df["Humidity"].min():.2f}% to {df["Humidity"].max():.2f}%')
    print(f'  CO2_InfraredSensor min after fix: {df["CO2_InfraredSensor"].min():.2f} ppm')
    print(f'  Remaining nulls                 : {df.isnull().sum().sum()}')

    return df

# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Univariate analysis
# ─────────────────────────────────────────────────────────────────────────────
# Function for displaying the data and any features individually
def univariate_analysis(df: pd.DataFrame) -> pd.DataFrame:

    "Print summary statistics and plot distributions for all features."

    # Summary stats
    desc = df[NUMERIC_COLS].describe().T
    desc['skewness'] = df[NUMERIC_COLS].skew().round(3)
    desc['kurtosis'] = df[NUMERIC_COLS].kurt().round(3)
    print(desc.round(3).to_string())

    # Skewness/Kurtosis interpretation
    # Skew > 0.5 = right-skewed (use log1p transform for linear models)
    # Skew < -0.5 = left-skewed
    # Kurt > 1 = leptokurtic = more extreme outliers than normal
    print('\n=== Skewness & Kurtosis Notes ===')\
    
    # For loop: prints the skewness and kurtosis interpretation for each numeric column
    for col in NUMERIC_COLS:
        skew = df[col].skew()
        kurt = df[col].kurt()

        # If statement: flags any columns that needs log transform (skew >= 0.5)
        if abs(skew) >= 0.5:
            print(f'  {col}: skew={skew:.2f} — consider log1p transform for linear models')
        if kurt > 1:
            print(f'  {col}: kurt={kurt:.2f} — heavy tails, more outliers than normal distribution')

    # Numeric distributions
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    # For loop: plots a histogram with KDE curve for every numeric sensor
    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i]
        sns.histplot(df[col], bins=50, kde=True, ax=ax, color='#4C72B0', alpha=0.7)
        ax.set_title(f'Graph {i+1} - {col}', fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        skew_val = df[col].skew()
        kurt_val = df[col].kurt()

        # Colours the graph annotation red if skew >= 0.5
        colour = '#c0392b' if abs(skew_val) >= 0.5 else '#27ae60'
        ax.annotate(f'skew={skew_val:.2f}  kurt={kurt_val:.2f}',
                    xy=(0.97, 0.92), xycoords='axes fraction',
                    ha='right', fontsize=8, color=colour)
        
    # Plotting
    plt.suptitle('Distribution of Numeric Sensor Features (post-cleaning)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    # Categorical distributions
    # List the columns that needs to be visualized
    cat_cols = ['Activity Level', 'Time of Day', 'HVAC Operation Mode', 'Ambient Light Level']
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, col in enumerate(cat_cols):
        ax = axes[i]

        # Counts the amount of times the category appeards
        vc = df[col].value_counts()
        vc.plot(kind='bar', ax=ax, color=sns.color_palette('muted', len(vc)), edgecolor='white')
        ax.set_title(f'Graph {i+10} - {col}', fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)

        # For loop: adds count label on top of each bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2, p.get_height() + 10),
                        ha='center', fontsize=8)
            
    # Plotting
    plt.suptitle('Categorical Feature Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return desc.round(3)


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — Bivariate analysis
# ─────────────────────────────────────────────────────────────────────────────
# Function for comparison of data to the activity level
def bivariate_analysis(df: pd.DataFrame) -> pd.DataFrame:

    "Visualise how each feature relates to Activity Level via using different types of visualizations"

    # Box plots — all numeric sensors
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes = axes.flatten()

    # For loop: plotting a box plot for the numeric sensor against the activity level
    for i, col in enumerate(NUMERIC_COLS):
        sns.boxplot(
            data=df, x='Activity Level', y=col, order=ACTIVITY_ORDER,
            palette=PALETTE, ax=axes[i], width=0.5,

            # controls how any outlier dots look like on the box plot
            flierprops=dict(marker='o', markersize=2, alpha=0.3),
        )
        axes[i].set_title(f'Graph {i+14} - {col}', fontweight='bold', fontsize=10)        
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=15)

    # Plotting
    plt.suptitle('Sensor Readings by Activity Level', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    # Violin plots — CO2 sensors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # For loop: plots for C)2 sensors to compare their distribution
    for i, col in enumerate(['CO2_InfraredSensor', 'CO2_ElectroChemicalSensor']):
        sns.violinplot(
            data=df, x='Activity Level', y=col, order=ACTIVITY_ORDER,
            palette=PALETTE, ax=axes[i], inner='quartile',
        )
        axes[i].set_title(f'Graph {i+23} - {col} by Activity Level', fontweight='bold')
        axes[i].set_xlabel('')

    # Plotting
    plt.suptitle('CO2 Sensor Distributions by Activity Level', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Stacked bar — categorical features
    # List the columns that will be compared against the activity level
    cat_features = ['Time of Day', 'HVAC Operation Mode', 'Ambient Light Level']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # For loop: plotting a normalized stacked bar chart for the categorical features
    for i, col in enumerate(cat_features):

        # Counts how many times the category appears with each activity
        ct = pd.crosstab(df[col], df['Activity Level'], normalize='index') * 100

        # Keeps only the activity columns that exists
        ct = ct[[c for c in ACTIVITY_ORDER if c in ct.columns]]
        ct.plot(kind='bar', stacked=True, ax=axes[i],
                color=[PALETTE[c] for c in ct.columns], edgecolor='white', width=0.7)
        axes[i].set_title(f'Graph {i+25} - {col} vs Activity Level', fontweight='bold')
        axes[i].set_ylabel('% of rows')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=35)
        axes[i].legend(loc='upper right', fontsize=8)

    # Plotting
    plt.suptitle('Categorical Features vs Activity Level (normalised)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Group means table
    group_means = df.groupby('Activity Level')[NUMERIC_COLS].mean().round(2)

    return group_means.loc[[a for a in ACTIVITY_ORDER if a in group_means.index]]


# ─────────────────────────────────────────────────────────────────────────────
# PART 6 — Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────
# Function to show which features are strongly correlated
def correlation_analysis(df: pd.DataFrame) -> pd.Series:
    
    "Compute and visualise Pearson correlations among numeric features and between features and the ordinally-encoded target."

    # Maps the activity classes into correlation so that it can be calculated
    activity_encode = {'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}
    df = df.copy()
    df['Activity_Encoded'] = df['Activity Level'].map(activity_encode)

    # Combines the numeric sensor columns with the encoded target for correlation
    corr_cols = NUMERIC_COLS + ['Activity_Encoded']

    # Computing the Pearson correlation between every pair of columns
    corr_matrix = df[corr_cols].corr()

    # Full heatmap
    fig, ax = plt.subplots(figsize=(13, 10))

    # Hides the upper triabgle to avoud showing any duplicate values
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, ax=ax,
        linewidths=0.5, annot_kws={'size': 9},
    )
    ax.set_title('Graph 28 - Pearson Correlation Matrix (Numeric Features + Encoded Target)',
                 fontweight='bold', pad=15)
    
    # Plotting
    plt.tight_layout()
    plt.show()

    # Feature-target correlations ranked
    # Dropts the target column itself and sorts the features by correlation
    target_corr = corr_matrix['Activity_Encoded'].drop('Activity_Encoded').abs().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    target_corr.plot(kind='bar', ax=ax, color=sns.color_palette('Blues_r', len(target_corr)))
    ax.set_title('Graph 29 - Feature Correlation with Activity Level (absolute Pearson r)', fontweight='bold')
    ax.set_ylabel('|Pearson r|')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=35)

    # For liip: adds the correlation value on the bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}',
                    (p.get_x() + p.get_width() / 2, p.get_height() + 0.002),
                    ha='center', fontsize=9)
        
    # Plotting
    plt.tight_layout()
    plt.show()

    return target_corr


# ─────────────────────────────────────────────────────────────────────────────
# PART 7 — Session-level analysis
# ─────────────────────────────────────────────────────────────────────────────
# Function for analysing and examingin thow the sensor readings and activities can vary across the different types of session
def session_analysis(df: pd.DataFrame) -> pd.DataFrame:

    "Analyse how sensor readings and activity distributions vary across sessions."

    activity_encode = {'Low Activity': 0, 'Moderate Activity': 1, 'High Activity': 2}
    df = df.copy()
    df['Activity_Encoded'] = df['Activity Level'].map(activity_encode)

    # Groups the data based on their Session ID and calculates summary statistics for each session
    # Lambda functions: calculate the percentage of each activity class by the session
    session_stats = df.groupby('Session ID').agg(
        rows=('Activity Level', 'count'), # Total rows per session
        low_pct=('Activity_Encoded', lambda x: (x == 0).mean() * 100), # % of Low Activity
        mod_pct=('Activity_Encoded', lambda x: (x == 1).mean() * 100), # % of Moderate Activity
        high_pct=('Activity_Encoded', lambda x: (x == 2).mean() * 100), # % of High Activity
        avg_co2=('CO2_ElectroChemicalSensor', 'mean'), # Mean C02 per session
        avg_temp=('Temperature', 'mean'), # mean teperature per session
    ).reset_index() # converts the Session ID back into a regular column

    print(f'Unique sessions : {df["Session ID"].nunique()}')
    print(f'Rows per session — min: {session_stats["rows"].min()}, '
          f'max: {session_stats["rows"].max()}, '
          f'mean: {session_stats["rows"].mean():.1f}')

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    session_stats['rows'].plot(kind='hist', bins=30, ax=axes[0], color='#4C72B0', edgecolor='white')
    axes[0].set_title('Graph 30 - Distribution of Rows per Session', fontweight='bold')
    axes[0].set_xlabel('Number of rows')

    axes[1].scatter(session_stats['Session ID'], session_stats['avg_co2'],
                    alpha=0.5, s=15, color='#DD8452')
    axes[1].set_title('Graph 31 - Mean CO2 (ElectroChemical) per Session', fontweight='bold')
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

    # Plotting
    ax.set_title('Graph 32 - Activity Level Distribution across Sessions (top 20)', fontweight='bold')
    ax.set_xlabel('Session ID')
    ax.set_ylabel('% of readings')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return session_stats


# ─────────────────────────────────────────────────────────────────────────────
# PART 8 — Summary
# ─────────────────────────────────────────────────────────────────────────────
# Function to display and print the summary
def summary(df: pd.DataFrame, save_csv: bool = True) -> None:

    "Print the final cleaned dataset summary and optionally save cleaned CSV."

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

