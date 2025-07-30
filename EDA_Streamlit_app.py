import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page config
st.set_page_config(
    page_title="Volleyball Nations League 2023 EDA",
    page_icon="🏐",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('VNL2023.csv')

df = load_data()

# Title and description
st.title("🏐 Volleyball Nations League 2023 - Player Statistics")
st.markdown("""
Exploratory Data Analysis of player performance statistics from the 2023 Volleyball Nations League.
""")

# Sidebar filters
st.sidebar.header("Filters")
selected_positions = st.sidebar.multiselect(
    "Select positions to include",
    options=df['Position'].unique(),
    default=df['Position'].unique()
)

age_range = st.sidebar.slider(
    "Select age range",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

# Apply filters
filtered_df = df[
    (df['Position'].isin(selected_positions)) & 
    (df['Age'].between(age_range[0], age_range[1]))
]

# Show filtered data
st.sidebar.markdown(f"**Filtered Players:** {len(filtered_df)} of {len(df)}")

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Position Distribution", 
    "Skills by Position",
    "Age Analysis",
    "Top Attackers",
    "Skill Correlations",
    "Country Performance"
])

# Tab 1: Position Distribution
with tab1:
    st.header("Player Position Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    position_counts = filtered_df['Position'].value_counts()
    sns.barplot(x=position_counts.index, y=position_counts.values, palette="viridis", ax=ax)
    plt.title('Distribution of Players by Position', fontsize=16)
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Number of Players', fontsize=14)
    st.pyplot(fig)
    
    with st.expander("View position descriptions"):
        st.markdown("""
        - **OH**: Outside Hitter (primary attackers who also play defense)
        - **OP**: Opposite (typically the main attacker, doesn't receive serves)
        - **MB**: Middle Blocker (specialize in blocking and quick attacks)
        - **S**: Setter (team's playmaker who sets up attacks)
        - **L**: Libero (defensive specialist who can't attack)
        """)

# Tab 2: Skills by Position
with tab2:
    st.header("Average Skill Ratings by Position")
    skills = ['Attack', 'Block', 'Serve', 'Set', 'Dig', 'Receive']
    
    # Let user select which skills to show
    selected_skills = st.multiselect(
        "Select skills to display",
        options=skills,
        default=skills
    )
    
    if selected_skills:
        fig, ax = plt.subplots(figsize=(12, 8))
        position_skills = filtered_df.groupby('Position')[selected_skills].mean().transpose()
        sns.heatmap(position_skills, annot=True, cmap="YlGnBu", fmt=".1f", linewidths=.5, ax=ax)
        plt.title('Average Skill Ratings by Position', fontsize=16)
        plt.xlabel('Position', fontsize=14)
        plt.ylabel('Skill', fontsize=14)
        st.pyplot(fig)
    else:
        st.warning("Please select at least one skill to display")

# Tab 3: Age Analysis
with tab3:
    st.header("Age Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution by Position")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Position', y='Age', data=filtered_df, palette="Set2", ax=ax)
        plt.title('Age Distribution by Position', fontsize=16)
        plt.xlabel('Position', fontsize=14)
        plt.ylabel('Age', fontsize=14)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Age vs. Attack Performance")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Age', y='Attack', hue='Position', data=filtered_df, palette="Set2", s=100, ax=ax)
        plt.title('Attack Performance by Age', fontsize=16)
        plt.xlabel('Age', fontsize=14)
        plt.ylabel('Attack Score', fontsize=14)
        st.pyplot(fig)

# Tab 4: Top Attackers
with tab4:
    st.header("Top Attacking Players")
    
    top_n = st.slider("Select number of top players to show", 5, 20, 10)
    
    top_attack = filtered_df.nlargest(top_n, 'Attack')[['Player', 'Country', 'Attack', 'Position', 'Age']]
    
    # Show as both table and chart
    st.dataframe(top_attack.style.background_gradient(subset=['Attack'], cmap='YlOrRd'))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Attack', y='Player', hue='Position', data=top_attack, palette="viridis", dodge=False, ax=ax)
    plt.title(f'Top {top_n} Players by Attack Score', fontsize=16)
    plt.xlabel('Attack Score', fontsize=14)
    plt.ylabel('Player', fontsize=14)
    st.pyplot(fig)

# Tab 5: Skill Correlations
with tab5:
    st.header("Skill Correlations")
    
    skills = ['Attack', 'Block', 'Serve', 'Set', 'Dig', 'Receive', 'Age']
    selected_corr_skills = st.multiselect(
        "Select skills for correlation analysis",
        options=skills,
        default=['Attack', 'Block', 'Serve', 'Dig']
    )
    
    if len(selected_corr_skills) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = filtered_df[selected_corr_skills].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=.5, ax=ax)
        plt.title('Correlation Between Selected Skills', fontsize=16)
        st.pyplot(fig)
        
        with st.expander("Interpretation Guide"):
            st.markdown("""
            - **+1.0**: Perfect positive correlation
            - **+0.5 to +0.9**: Strong positive relationship
            - **0 to +0.5**: Weak positive relationship
            - **0**: No correlation
            - **Negative values**: Inverse relationship
            """)
    else:
        st.warning("Please select at least 2 skills to see correlations")

# Tab 6: Country Performance
with tab6:
    st.header("Country Performance Analysis")
    
    min_players = st.slider("Minimum players per country to include", 1, 10, 3)
    
    country_stats = filtered_df.groupby('Country').agg({
        'Player': 'count', 
        'Attack': 'mean',
        'Block': 'mean',
        'Age': 'mean'
    }).sort_values('Player', ascending=False)
    country_stats = country_stats[country_stats['Player'] >= min_players]
    
    if not country_stats.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Country Representation")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=country_stats.index, y=country_stats['Player'], hue=country_stats.index, palette="viridis", legend=False, ax=ax)
            plt.title(f'Players per Country (min {min_players} players)', fontsize=16)
            plt.xlabel('Country', fontsize=14)
            plt.ylabel('Number of Players', fontsize=14)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Average Attack by Country")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=country_stats.index, y=country_stats['Attack'], hue=country_stats.index, palette="rocket", legend=False, ax=ax)
            plt.title('Average Attack Score by Country', fontsize=16)
            plt.xlabel('Country', fontsize=14)
            plt.ylabel('Average Attack Score', fontsize=14)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        st.subheader("Detailed Country Statistics")
        st.dataframe(country_stats.style.background_gradient(subset=['Attack', 'Block'], cmap='YlOrRd'))
    else:
        st.warning(f"No countries have at least {min_players} players with current filters")

# Add footer
st.markdown("---")
st.markdown("""
**Data Source**: Volleyball Nations League 2023 Player Statistics  
**Created with**: Python, Streamlit, Pandas, Seaborn  
""")

# Close all matplotlib figures to prevent memory leaks
plt.close('all')