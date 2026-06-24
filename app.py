import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

sns.set_theme(style='darkgrid', palette='muted')

st.set_page_config(
    page_title='FIFA 26 World Cup Simulator',
    page_icon='⚽',
    layout='wide'
)

# ============================================
# ACTUAL SCORES (update as matches are played)
# ============================================

actual_scores = {
    # ('Home Team', 'Away Team'): (home_goals, away_goals)
    ('Mexico', 'South Africa'): (2, 0),
    ('Czech Republic', 'South Korea'): (1, 2),
    ('Canada', 'Bosnia and Herzegovina'): (1, 1),
    ('United States', 'Paraguay'): (4, 1),
    ('Qatar', 'Switzerland'): (1, 1),
    ('Brazil', 'Morocco'): (1, 1),
    ('Haiti', 'Scotland'): (0, 1),
    ('Australia', 'Turkey'): (2, 0),
    ('Germany', 'Curaçao'): (7, 1),
    ('Netherlands', 'Japan'): (2, 2),
    ('Ivory Coast', 'Ecuador'): (1, 0),
    ('Sweden', 'Tunisia'): (5, 1),
    ('Spain', 'Cape Verde'): (0, 0),
    ('Belgium', 'Egypt'): (1, 1),
    ('Saudi Arabia', 'Uruguay'): (1, 1),
    ('Iran', 'New Zealand'): (2, 2),
    ('France', 'Senegal'): (3, 1),
    ('Iraq', 'Norway'): (1, 4),
    ('Argentina', 'Algeria'): (3, 0),
    ('Austria', 'Jordan'): (3, 1),
    ('Portugal', 'DR Congo'): (1, 1),
    ('England', 'Croatia'): (4, 2),
    ('Ghana', 'Panama'): (1, 0),
    ('Uzbekistan', 'Colombia'): (1, 3),
    ('Czech Republic', 'South Africa'): (1, 1),
    ('Switzerland', 'Bosnia and Herzegovina'): (4, 1),
    ('Canada', 'Qatar'): (6, 0),
    ('Mexico', 'South Korea'): (1, 0),
    ('United States', 'Australia'): (2, 0),
    ('Scotland', 'Morocco'): (0, 1),
    ('Brazil', 'Haiti'): (3, 0),
    ('Turkey', 'Paraguay'): (0, 1),
    ('Netherlands', 'Sweden'): (5, 1),
    ('Germany', 'Ivory Coast'): (2, 1),
    ('Ecuador', 'Curaçao'): (0, 0),
    ('Tunisia', 'Japan'): (0, 4),
    ('Spain', 'Saudi Arabia'): (4, 0),
    ('Belgium', 'Iran'): (0, 0),
    ('Uruguay', 'Cape Verde'): (2, 2),
    ('New Zealand', 'Egypt'): (1, 3),
    ('Argentina', 'Austria'): (2, 0),
    ('France', 'Iraq'): (3, 0),
    ('Norway', 'Senegal'): (3, 2),
    ('Jordan', 'Algeria'): (1, 2),
    ('Portugal', 'Uzbekistan'): (5, 0),
    ('England', 'Ghana'): (0, 0),
    ('Panama', 'Croatia'): (0, 1),
    ('Colombia', 'DR Congo'): (1, 0),
}

# ============================================
# GROUPS
# ============================================

groups = {
    'A': ['Mexico', 'South Africa', 'Czech Republic', 'South Korea'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

# ============================================
# CACHED DATA LOADING
# ============================================

@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    elo_df = pd.read_csv('elo_ratings.csv')
    elo_ratings = dict(zip(elo_df['team'], elo_df['elo']))
    rankings_df = pd.read_csv('fifa_rankings.csv')
    prob_df = pd.read_csv('advancement_probabilities.csv')
    fixtures_df = pd.read_csv('monte_carlo_fixtures.csv')
    return df, elo_ratings, rankings_df, prob_df, fixtures_df

@st.cache_resource
def train_model(df):
    features = [
        'elo_difference', 'home_elo', 'away_elo', 'neutral',
        'tournament_weight', 'home_form', 'away_form',
        'home_ranking', 'away_ranking', 'ranking_difference'
    ]
    X = df[features]
    y = df['result']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        min_samples_leaf=5, random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

# ============================================
# KNOCKOUT STAGE — DATA + HELPERS
# ============================================

# --------------------------------------------
# REAL ROUND OF 32 BRACKET
# --------------------------------------------
# Fill this in once the actual Round of 32 matchups are confirmed.
# Order matters: pairs (0,1), (2,3), (4,5)... feed into the next round
# in that order. Leave as [] until the bracket is set — the knockout
# pages below will use mock data automatically until this is populated.

round_of_32_real = [
    # fill in from main.py
]

ROUND_ORDER = ['Round of 32', 'Round of 16', 'Quarterfinals', 'Semifinals', 'Final']

# --------------------------------------------
# REAL KNOCKOUT RESULTS (update as matches happen)
# --------------------------------------------
# Once an actual knockout match is played, add it here. Format matches
# actual_scores above. Any match in here is fully locked in the bracket
# simulator — no simulate button, just the real result shown.

knockout_actual_scores = {
    # ('Home Team', 'Away Team'): (home_goals, away_goals, 'H'/'D'/'A', went_to_pens, pen_home, pen_away)
    # Example: ('Mexico', 'Iran'): (1, 1, 'H', True, 4, 2),
}

def penalty_shootout_winner(home_elo, away_elo):
    elo_diff = home_elo - away_elo
    home_win_prob = np.clip(0.5 + 0.0003 * elo_diff, 0.35, 0.65)
    winner = 'H' if np.random.random() < home_win_prob else 'A'
    return winner, home_win_prob

def simulate_knockout_match(home_team, away_team, elo_ratings, comp_arrays):
    h, a, regulation_result, home_elo, away_elo = predict_score_and_result(
        home_team, away_team, elo_ratings, comp_arrays
    )
    went_to_penalties = False
    pen_home, pen_away = None, None

    if regulation_result == 'D':
        went_to_penalties = True
        winner, _ = penalty_shootout_winner(home_elo, away_elo)
        pen_home = np.random.randint(3, 6)
        pen_away = np.random.randint(3, 6)
        while pen_home == pen_away:
            pen_away = np.random.randint(3, 6)
        if winner == 'H' and pen_home < pen_away:
            pen_home, pen_away = pen_away, pen_home
        elif winner == 'A' and pen_away < pen_home:
            pen_home, pen_away = pen_away, pen_home
        advancing_team = home_team if winner == 'H' else away_team
    else:
        advancing_team = home_team if regulation_result == 'H' else away_team

    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_goals': int(h),
        'away_goals': int(a),
        'went_to_penalties': went_to_penalties,
        'pen_home': pen_home,
        'pen_away': pen_away,
        'advancing_team': advancing_team,
        'home_elo': home_elo,
        'away_elo': away_elo,
    }

def get_active_bracket():
    return round_of_32_real

def get_real_knockout_result(home, away):
    # Checks knockout_actual_scores for a real result, handling either team order.
    if (home, away) in knockout_actual_scores:
        hg, ag, result, pens, ph, pa = knockout_actual_scores[(home, away)]
        return {
            'home_team': home, 'away_team': away,
            'home_goals': hg, 'away_goals': ag,
            'went_to_penalties': pens, 'pen_home': ph, 'pen_away': pa,
            'advancing_team': home if result == 'H' else away,
            'is_real': True
        }
    if (away, home) in knockout_actual_scores:
        ag, hg, result, pens, pa, ph = knockout_actual_scores[(away, home)]
        # stored with away as "home" in the dict, so flip back
        return {
            'home_team': home, 'away_team': away,
            'home_goals': hg, 'away_goals': ag,
            'went_to_penalties': pens, 'pen_home': ph, 'pen_away': pa,
            'advancing_team': home if result == 'A' else away,
            'is_real': True
        }
    return None

def init_bracket_state(bracket):
    # Initializes session state for the user's personal simulated bracket. Only resets if the bracket size has changed (e.g. mock -> real swap), otherwise leaves existing kept results untouched across reruns.
    needs_init = (
        'bracket_state' not in st.session_state
        or len(st.session_state.bracket_state['Round of 32']) != len(bracket)
    )
    if needs_init:
        st.session_state.bracket_state = {
            'Round of 32': {i: None for i in range(len(bracket))},
            'Round of 16': {},
            'Quarterfinals': {},
            'Semifinals': {},
            'Final': {},
            'Third Place Playoff': {},
        }

def get_round_matchups(round_name, bracket):
    # Returns the list of (home, away) matchups for a round, using kept results from prior rounds where available, or None if not yet determined.
    state = st.session_state.bracket_state

    if round_name == 'Round of 32':
        return bracket

    prior_round = ROUND_ORDER[ROUND_ORDER.index(round_name) - 1]
    prior_matchups = get_round_matchups(prior_round, bracket)
    prior_results = state[prior_round]

    advancing = []
    for i in range(len(prior_matchups)):
        result = prior_results.get(i)
        advancing.append(result['advancing_team'] if result else None)

    matchups = []
    for i in range(0, len(advancing), 2):
        matchups.append((advancing[i], advancing[i + 1]))
    return matchups

# ============================================
# HELPERS
# ============================================

def get_recent_form(df, team, date, n=5):
    past = df[
        (df['date'] < date) &
        ((df['home_team'] == team) | (df['away_team'] == team))
    ].tail(n)
    if len(past) == 0:
        return 0.5
    points = 0
    for _, row in past.iterrows():
        if row['result'] == 'H' and row['home_team'] == team:
            points += 3
        elif row['result'] == 'A' and row['away_team'] == team:
            points += 3
        elif row['result'] == 'D':
            points += 1
    return points / (n * 3)

def get_competitive_arrays(df):
    competitive = df[df['tournament_weight'] >= 2].copy()
    return (
        competitive['home_score'].values,
        competitive['away_score'].values,
        competitive['elo_difference'].values,
        competitive['result'].values,
    )

def predict_score_and_result(home_team, away_team, elo_ratings, comp_arrays):
    home_scores, away_scores, elo_diffs, _ = comp_arrays
    home_elo = elo_ratings.get(home_team, 1000)
    away_elo = elo_ratings.get(away_team, 1000)
    elo_diff = home_elo - away_elo

    sigma = 100
    elo_similarity = np.exp(-((elo_diffs - elo_diff) ** 2) / (2 * sigma ** 2))
    weights = elo_similarity / elo_similarity.sum()

    avg_home = np.dot(home_scores, weights)
    avg_away = np.dot(away_scores, weights)

    elo_factor = np.clip(elo_diff / 400, -1.0, 1.0)
    avg_home *= (1 + 0.35 * elo_factor)
    avg_away *= (1 - 0.35 * elo_factor)

    avg_home += np.random.normal(0, 0.15)
    avg_away += np.random.normal(0, 0.15)

    avg_home = max(0, min(avg_home, 2.3))
    avg_away = max(0, min(avg_away, 2.3))

    h = np.random.poisson(avg_home)
    a = np.random.poisson(avg_away)

    if h > a:
        result = 'H'
    elif a > h:
        result = 'A'
    else:
        result = 'D'

    return int(h), int(a), result, home_elo, away_elo

def simulate_group(group_name, predictions_df):
    group_matches = predictions_df[predictions_df['group'] == group_name]
    teams = groups[group_name]
    standings = {
        team: {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0}
        for team in teams
    }
    for _, match in group_matches.iterrows():
        home = match['home_team']
        away = match['away_team']
        hg = int(match['home_goals'])
        ag = int(match['away_goals'])
        result = match['predicted_result']

        standings[home]['P'] += 1
        standings[away]['P'] += 1
        standings[home]['GF'] += hg
        standings[home]['GA'] += ag
        standings[away]['GF'] += ag
        standings[away]['GA'] += hg

        if result == 'H':
            standings[home]['W'] += 1
            standings[home]['Pts'] += 3
            standings[away]['L'] += 1
        elif result == 'A':
            standings[away]['W'] += 1
            standings[away]['Pts'] += 3
            standings[home]['L'] += 1
        else:
            standings[home]['D'] += 1
            standings[away]['D'] += 1
            standings[home]['Pts'] += 1
            standings[away]['Pts'] += 1

    standings_df = pd.DataFrame(standings).T
    standings_df['GD'] = standings_df['GF'] - standings_df['GA']
    standings_df = standings_df.sort_values(['Pts', 'GD', 'GF'], ascending=False)
    standings_df.index.name = 'Team'
    return standings_df[['P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']]

# ============================================
# LOAD DATA + TRAIN MODEL
# ============================================

df, elo_ratings, rankings_df, prob_df, fixtures_df = load_data()
model = train_model(df)
comp_arrays = get_competitive_arrays(df)

# ============================================
# UI — SIDEBAR
# ============================================

st.sidebar.title('⚽ FIFA 26 Simulator')
st.sidebar.markdown('---')

section = st.sidebar.radio('Section', ['Group Stage', 'Knockout Stage'], horizontal=True, label_visibility='collapsed')
st.sidebar.markdown('---')

if section == 'Group Stage':
    page = st.sidebar.radio(
        'Group stage view',
        ['Match Simulator', 'Group Standings', 'Advancement Probabilities', 'All Match Results'],
        label_visibility='collapsed'
    )
else:
    page = st.sidebar.radio(
        'Knockout stage view',
        ['Bracket Simulator', 'Advancement Probabilities (Knockout)'],
        label_visibility='collapsed'
    )

st.sidebar.markdown('---')
st.sidebar.caption('Model: Random Forest + Poisson')
st.sidebar.caption('Monte Carlo: 1,000 simulations')

# ============================================
# PAGE 1 — MATCH SIMULATOR
# ============================================

if page == 'Match Simulator':
    st.title('Match Simulator')
    st.markdown('Pick any group stage matchup, run a live simulation, and compare it against the Monte Carlo consensus.')

    col_sel1, col_sel2, col_sel3 = st.columns(3)

    with col_sel1:
        selected_group = st.selectbox('Group', sorted(groups.keys()), format_func=lambda g: f'Group {g}')

    group_teams = groups[selected_group]

    with col_sel2:
        home_team = st.selectbox('Home team', group_teams)

    away_options = [t for t in group_teams if t != home_team]
    with col_sel3:
        away_team = st.selectbox('Away team', away_options)

    st.markdown('---')

    # Check for actual result
    actual_key = (home_team, away_team)
    actual_key_rev = (away_team, home_team)
    actual = actual_scores.get(actual_key) or actual_scores.get(actual_key_rev)

    if actual:
        if actual_scores.get(actual_key):
            hg_actual, ag_actual = actual
        else:
            ag_actual, hg_actual = actual
        st.info(
            f'🟢 **Actual result played:** {home_team} **{hg_actual} – {ag_actual}** {away_team}'
        )

    # Run simulation
    if 'sim_result' not in st.session_state:
        st.session_state.sim_result = None
    if 'sim_key' not in st.session_state:
        st.session_state.sim_key = None

    current_key = (home_team, away_team)
    if st.session_state.sim_key != current_key:
        st.session_state.sim_result = None

    if st.button('▶ Simulate this match', use_container_width=True):
        h, a, result, home_elo, away_elo = predict_score_and_result(
            home_team, away_team, elo_ratings, comp_arrays
        )
        st.session_state.sim_result = (h, a, result, home_elo, away_elo)
        st.session_state.sim_key = current_key

    col_live, col_mc = st.columns(2)

    with col_live:
        st.subheader('Live simulation')
        if st.session_state.sim_result:
            h, a, result, home_elo, away_elo = st.session_state.sim_result
            elo_diff = home_elo - away_elo

            if result == 'H':
                winner_text = f'{home_team} win'
                home_color = 'green'
                away_color = 'red'
            elif result == 'A':
                winner_text = f'{away_team} win'
                home_color = 'red'
                away_color = 'green'
            else:
                winner_text = 'Draw'
                home_color = away_color = 'gray'

            st.markdown(
                f"""
                <div style="background:#f8f9fa;border-radius:12px;padding:24px;text-align:center;border:1px solid #e0e0e0">
                    <div style="display:flex;justify-content:space-around;align-items:center">
                        <div>
                            <div style="font-size:14px;color:#666;margin-bottom:4px">{home_team}</div>
                            <div style="font-size:48px;font-weight:700;color:{'#1a6b3c' if home_color=='green' else '#c0392b' if home_color=='red' else '#555'}">{h}</div>
                            <div style="font-size:11px;color:#999">ELO: {home_elo:.0f}</div>
                        </div>
                        <div style="font-size:24px;color:#999">—</div>
                        <div>
                            <div style="font-size:14px;color:#666;margin-bottom:4px">{away_team}</div>
                            <div style="font-size:48px;font-weight:700;color:{'#1a6b3c' if away_color=='green' else '#c0392b' if away_color=='red' else '#555'}">{a}</div>
                            <div style="font-size:11px;color:#999">ELO: {away_elo:.0f}</div>
                        </div>
                    </div>
                    <div style="margin-top:16px;font-size:13px;color:#444;font-weight:500">{winner_text}</div>
                    <div style="font-size:11px;color:#999;margin-top:4px">ELO diff: {elo_diff:+.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button('↻ Simulate again'):
                h, a, result, home_elo, away_elo = predict_score_and_result(
                    home_team, away_team, elo_ratings, comp_arrays
                )
                st.session_state.sim_result = (h, a, result, home_elo, away_elo)
                st.session_state.sim_key = current_key
                st.rerun()
        else:
            st.markdown(
                '<div style="background:#f8f9fa;border-radius:12px;padding:40px;text-align:center;color:#999;border:1px solid #e0e0e0">Press simulate to run a prediction</div>',
                unsafe_allow_html=True
            )

    with col_mc:
        st.subheader('Monte Carlo consensus')

        mc_match = fixtures_df[
            (fixtures_df['home_team'] == home_team) &
            (fixtures_df['away_team'] == away_team)
        ]
        if mc_match.empty:
            mc_match = fixtures_df[
                (fixtures_df['home_team'] == away_team) &
                (fixtures_df['away_team'] == home_team)
            ]
            flipped = True
        else:
            flipped = False

        if not mc_match.empty:
            row = mc_match.iloc[0]
            mc_h = int(row['home_goals']) if not flipped else int(row['away_goals'])
            mc_a = int(row['away_goals']) if not flipped else int(row['home_goals'])
            mc_result = row['predicted_result']

            if (not flipped and mc_result == 'H') or (flipped and mc_result == 'A'):
                mc_winner = f'{home_team} win'
            elif (not flipped and mc_result == 'A') or (flipped and mc_result == 'H'):
                mc_winner = f'{away_team} win'
            else:
                mc_winner = 'Draw'

            st.markdown(
                f"""
                <div style="background:#f0f4ff;border-radius:12px;padding:24px;text-align:center;border:1px solid #c7d2f0">
                    <div style="display:flex;justify-content:space-around;align-items:center">
                        <div>
                            <div style="font-size:14px;color:#666;margin-bottom:4px">{home_team}</div>
                            <div style="font-size:48px;font-weight:700;color:#2c3e7a">{mc_h}</div>
                        </div>
                        <div style="font-size:24px;color:#999">—</div>
                        <div>
                            <div style="font-size:14px;color:#666;margin-bottom:4px">{away_team}</div>
                            <div style="font-size:48px;font-weight:700;color:#2c3e7a">{mc_a}</div>
                        </div>
                    </div>
                    <div style="margin-top:16px;font-size:13px;color:#444;font-weight:500">{mc_winner}</div>
                    <div style="font-size:11px;color:#888;margin-top:4px">Most common across 1,000 simulations</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Per-match win probabilities from Monte Carlo
            if all(c in fixtures_df.columns for c in ['home_win%', 'draw%', 'away_win%']):
                hw = float(row['home_win%']) if not flipped else float(row['away_win%'])
                dr = float(row['draw%'])
                aw = float(row['away_win%']) if not flipped else float(row['home_win%'])

                st.markdown('#### Match probabilities (1,000 simulations)')

                bar_data = [
                    (f'{home_team} win', hw, '#1a6b3c'),
                    ('Draw',             dr, '#888888'),
                    (f'{away_team} win', aw, '#c0392b'),
                ]

                fig, ax = plt.subplots(figsize=(6, 2.2))
                outcomes = [b[0] for b in bar_data]
                values   = [b[1] for b in bar_data]
                colors_b = [b[2] for b in bar_data]

                bars = ax.barh(outcomes, values, color=colors_b, height=0.5)

                for bar, val in zip(bars, values):
                    if val >= 8:
                        ax.text(
                            bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            f'{val:.0f}%', ha='center', va='center',
                            fontsize=10, color='white', fontweight='bold'
                        )
                    else:
                        ax.text(
                            bar.get_width() + 1,
                            bar.get_y() + bar.get_height() / 2,
                            f'{val:.0f}%', ha='left', va='center',
                            fontsize=10, color='#444'
                        )

                ax.set_xlim(0, 100)
                ax.set_xlabel('Probability (%)', fontsize=9)
                ax.tick_params(axis='y', labelsize=10)
                ax.tick_params(axis='x', labelsize=8)
                sns.despine(left=False, bottom=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.warning('No Monte Carlo data found for this matchup.')

# ============================================
# PAGE 2 — GROUP STANDINGS
# ============================================

elif page == 'Group Standings':
    st.title('Group Stage Standings')
    st.markdown('Standings based on Monte Carlo consensus match results.')

    selected_group = st.selectbox('Select group', sorted(groups.keys()), format_func=lambda g: f'Group {g}')

    standings = simulate_group(selected_group, fixtures_df)

    st.markdown(f'### Group {selected_group}')

    def highlight_standings(row):
        idx = standings.index.get_loc(row.name)
        if idx == 0:
            return ['background-color: #d4edda; color: black'] * len(row)
        elif idx == 1:
            return ['background-color: #d4edda; color: black'] * len(row)
        elif idx == 2:
            return ['background-color: #fff3cd; color: black'] * len(row)
        else:
            return ['background-color: #f8d7da; color: black'] * len(row)

    st.dataframe(
        standings.style.apply(highlight_standings, axis=1),
        use_container_width=True
    )

    st.markdown('''
    🟢 **1st / 2nd** — Advance automatically  
    🟡 **3rd** — May advance as one of the best third-place teams  
    🔴 **4th** — Eliminated
    ''')

    st.markdown('---')
    st.subheader('All group standings')

    cols = st.columns(2)
    for i, group_name in enumerate(sorted(groups.keys())):
        with cols[i % 2]:
            st.markdown(f'**Group {group_name}**')
            s = simulate_group(group_name, fixtures_df)

            def highlight_all(row, s=s):
                idx = s.index.get_loc(row.name)
                if idx <= 1:
                    return ['background-color: #d4edda; color: black'] * len(row)
                elif idx == 2:
                    return ['background-color: #fff3cd; color: black'] * len(row)
                else:
                    return ['background-color: #f8d7da; color: black'] * len(row)

            st.dataframe(s.style.apply(highlight_all, axis=1), use_container_width=True)

# ============================================
# PAGE 3 — ADVANCEMENT PROBABILITIES
# ============================================

elif page == 'Advancement Probabilities':
    st.title('Advancement Probabilities')
    st.markdown('Based on 1,000 Monte Carlo simulations of the group stage.')

    selected_group = st.selectbox('Select group', sorted(groups.keys()), format_func=lambda g: f'Group {g}')

    group_prob = prob_df[prob_df['group'] == selected_group].copy()
    group_prob = group_prob.sort_values('advancement%', ascending=True)

    st.markdown(f'### Group {selected_group}')

    cols = ['1st%', '2nd%', '3rd_qual%', '3rd_elim%', '4th%']
    colors = {
        '1st%': '#1a6b3c',
        '2nd%': '#16b71b',
        '3rd_qual%': '#28f508',
        '3rd_elim%': '#e30909',
        '4th%': '#ff4444'
    }
    labels = {
        '1st%': '1st place',
        '2nd%': '2nd place',
        '3rd_qual%': '3rd (qualify)',
        '3rd_elim%': '3rd (eliminated)',
        '4th%': '4th place'
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    teams = group_prob['team'].tolist()
    left = np.zeros(len(teams))

    for col in ['1st%', '2nd%', '3rd_qual%', '3rd_elim%', '4th%']:
        values = group_prob[col].values
        bars = ax.barh(teams, values, left=left, color=colors[col], label=labels[col], height=0.6)
        for bar, val in zip(bars, values):
            if val >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.0f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold'
                )
        left += values

    ax.set_xlim(0, 100)
    ax.set_xlabel('Probability (%)')
    ax.axvline(x=50, color='white', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('---')
    st.subheader('Full probability table')
    display_cols = ['team', 'group', '1st%', '2nd%', '3rd_qual%', '3rd_elim%', '4th%', 'advancement%']
    st.dataframe(
        group_prob[display_cols].reset_index(drop=True),
        use_container_width=True
    )

# ============================================
# PAGE 4 — ALL MATCH RESULTS
# ============================================

elif page == 'All Match Results':
    st.title('All Predicted Match Results')
    st.markdown('Monte Carlo consensus scores for all 72 group stage matches.')

    selected_group = st.selectbox(
        'Filter by group',
        ['All'] + [f'Group {g}' for g in sorted(groups.keys())]
    )

    colors = {
        '1st%': '#1a6b3c', '2nd%': '#16b71b',
        '3rd_qual%': '#28f508', '3rd_elim%': '#e30909', '4th%': '#ff4444'
    }
    labels = {
        '1st%': '1st place', '2nd%': '2nd place',
        '3rd_qual%': '3rd (qualify)', '3rd_elim%': '3rd (eliminated)', '4th%': '4th place'
    }

    if selected_group == 'All':
        groups_to_show = sorted(groups.keys())
    else:
        groups_to_show = [selected_group.replace('Group ', '')]

    for group_name in groups_to_show:
        st.markdown(f'### Group {group_name}')
        group_matches = fixtures_df[fixtures_df['group'] == group_name].copy()
        group_matches = group_matches.reset_index(drop=True)

        for _, match in group_matches.iterrows():
            home = match['home_team']
            away = match['away_team']
            hg = int(match['home_goals'])
            ag = int(match['away_goals'])
            result = match['predicted_result']

            if result == 'H':
                home_color = '#1a6b3c'
                away_color = '#c0392b'
                label = f'🟡 {home} win'
            elif result == 'A':
                home_color = '#c0392b'
                away_color = '#1a6b3c'
                label = f'🟡 {away} win'
            else:
                home_color = away_color = '#666'
                label = '🟡 Draw'

            actual = actual_scores.get((home, away)) or actual_scores.get((away, home))
            actual_str = ''
            if actual:
                winner = f"{home} Wins" if (actual_scores.get((home, away)) and actual[0] > actual[1]) or (actual_scores.get((away, home)) and actual[1] > actual[0]) else f"{away} Wins" if (actual_scores.get((home, away)) and actual[1] > actual[0]) or (actual_scores.get((away, home)) and actual[0] > actual[1]) else 'Draw'
                if actual_scores.get((home, away)):
                    actual_str = f' &nbsp;|&nbsp; 🟢 Actual: {actual[0]}–{actual[1]} {winner}'
                else:
                    actual_str = f' &nbsp;|&nbsp; 🟢 Actual: {actual[1]}–{actual[0]} {winner}'

            st.markdown(
                f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                            padding:10px 16px;margin-bottom:6px;border-radius:8px;
                            background:#f9f9f9;border:1px solid #eee;font-size:14px">
                    <span style="color:{home_color};font-weight:{'600' if result=='H' else '400'};width:160px">{home}</span>
                    <span style="font-weight:700;font-size:16px;color:#222">{hg} — {ag}</span>
                    <span style="color:{away_color};font-weight:{'600' if result=='A' else '400'};width:160px;text-align:right">{away}</span>
                    <span style="color:#888;font-size:12px;width:160px;text-align:right">{label}{actual_str}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('')

# ============================================
# PAGE 5 — BRACKET SIMULATOR (KNOCKOUT STAGE)
# ============================================

elif page == 'Bracket Simulator':
    st.title('Knockout Bracket Simulator')

    active_bracket = get_active_bracket()
    using_mock = len(round_of_32_real) == 0

    if using_mock:
        st.warning('Round of 32 Matchups not yet finalized, stay tuned for updates!')

    st.markdown('Click into any matchup, simulate it, then **Keep** the result to lock it in and feed it into the next round. Your bracket is personal and resets if you reload the page.')

    init_bracket_state(active_bracket)

    if st.button('↺ Reset my bracket'):
        st.session_state.bracket_state = {
            'Round of 32': {i: None for i in range(len(active_bracket))},
            'Round of 16': {}, 'Quarterfinals': {}, 'Semifinals': {}, 'Final': {}, 'Third Place Playoff': {},
        }
        st.rerun()

    st.markdown('---')

    for round_name in ROUND_ORDER:
        matchups = get_round_matchups(round_name, active_bracket)
        st.subheader(round_name)

        if all(m[0] is None or m[1] is None for m in matchups) and round_name != 'Round of 32':
            st.caption('Complete the previous round to unlock these matchups.')

        for i, (home, away) in enumerate(matchups):
            if home is None or away is None:
                st.markdown(
                    f"""<div style="padding:10px 16px;margin-bottom:6px;border-radius:8px;
                    background:#f5f5f5;border:1px dashed #ccc;color:#999;font-size:14px">
                    TBD vs TBD</div>""",
                    unsafe_allow_html=True
                )
                continue

            real_result = get_real_knockout_result(home, away)
            state_key = f'{round_name}_{i}'

            with st.container(border=True):
                col_match, col_action = st.columns([3, 1])

                with col_match:
                    if real_result:
                        st.markdown(f'🔒 **{home}** {real_result["home_goals"]} — {real_result["away_goals"]} **{away}**')
                        if real_result['went_to_penalties']:
                            st.caption(f'Penalties: {real_result["pen_home"]}-{real_result["pen_away"]} · Actual result')
                        else:
                            st.caption('Actual result')
                        st.session_state.bracket_state[round_name][i] = real_result
                    else:
                        kept = st.session_state.bracket_state[round_name].get(i)
                        if kept:
                            pen_str = f' (PENS {kept["pen_home"]}-{kept["pen_away"]})' if kept['went_to_penalties'] else ''
                            st.markdown(f'**{home}** {kept["home_goals"]} — {kept["away_goals"]} **{away}**{pen_str}')
                            st.caption(f'✅ Kept — {kept["advancing_team"]} advances')
                        else:
                            st.markdown(f'**{home}** vs **{away}**')
                            st.caption('Not yet simulated')

                with col_action:
                    if not real_result:
                        if st.button('▶ Simulate', key=f'sim_{state_key}'):
                            result = simulate_knockout_match(home, away, elo_ratings, comp_arrays)
                            st.session_state[f'pending_{state_key}'] = result
                            st.rerun()

            pending = st.session_state.get(f'pending_{state_key}')
            if pending:
                pen_str = f' (PENS {pending["pen_home"]}-{pending["pen_away"]})' if pending['went_to_penalties'] else ''
                st.info(f'New simulation: **{home}** {pending["home_goals"]} — {pending["away_goals"]} **{away}**{pen_str} → **{pending["advancing_team"]}** advances')
                col_keep, col_resim = st.columns(2)
                with col_keep:
                    if st.button('✓ Keep this result', key=f'keep_{state_key}'):
                        st.session_state.bracket_state[round_name][i] = pending
                        del st.session_state[f'pending_{state_key}']
                        st.rerun()
                with col_resim:
                    if st.button('↻ Simulate again', key=f'resim_{state_key}'):
                        result = simulate_knockout_match(home, away, elo_ratings, comp_arrays)
                        st.session_state[f'pending_{state_key}'] = result
                        st.rerun()

        st.markdown('')

    # Third place playoff
    sf_matchups = get_round_matchups('Semifinals', active_bracket)
    sf_results = st.session_state.bracket_state['Semifinals']
    if len(sf_results) == 2 and all(sf_results.get(i) for i in range(2)):
        st.subheader('Third Place Playoff')
        losers = []
        for i in range(2):
            res = sf_results[i]
            loser = res['away_team'] if res['advancing_team'] == res['home_team'] else res['home_team']
            losers.append(loser)

        home3, away3 = losers[0], losers[1]
        real3 = get_real_knockout_result(home3, away3)
        kept3 = st.session_state.bracket_state['Third Place Playoff'].get(0)

        with st.container(border=True):
            col_match, col_action = st.columns([3, 1])
            with col_match:
                if real3:
                    st.markdown(f'🔒 **{home3}** {real3["home_goals"]} — {real3["away_goals"]} **{away3}**')
                    st.caption('Actual result')
                elif kept3:
                    pen_str = f' (PENS {kept3["pen_home"]}-{kept3["pen_away"]})' if kept3['went_to_penalties'] else ''
                    st.markdown(f'**{home3}** {kept3["home_goals"]} — {kept3["away_goals"]} **{away3}**{pen_str}')
                    st.caption(f'✅ Kept — {kept3["advancing_team"]} finishes 3rd')
                else:
                    st.markdown(f'**{home3}** vs **{away3}**')
                    st.caption('Not yet simulated')
            with col_action:
                if not real3 and st.button('▶ Simulate', key='sim_3rd'):
                    result = simulate_knockout_match(home3, away3, elo_ratings, comp_arrays)
                    st.session_state['pending_3rd'] = result
                    st.rerun()

        pending3 = st.session_state.get('pending_3rd')
        if pending3:
            pen_str = f' (PENS {pending3["pen_home"]}-{pending3["pen_away"]})' if pending3['went_to_penalties'] else ''
            st.info(f'New simulation: **{home3}** {pending3["home_goals"]} — {pending3["away_goals"]} **{away3}**{pen_str} → **{pending3["advancing_team"]}** finishes 3rd')
            col_keep, col_resim = st.columns(2)
            with col_keep:
                if st.button('✓ Keep this result', key='keep_3rd'):
                    st.session_state.bracket_state['Third Place Playoff'][0] = pending3
                    del st.session_state['pending_3rd']
                    st.rerun()
            with col_resim:
                if st.button('↻ Simulate again', key='resim_3rd'):
                    result = simulate_knockout_match(home3, away3, elo_ratings, comp_arrays)
                    st.session_state['pending_3rd'] = result
                    st.rerun()

    # Champion banner
    final_results = st.session_state.bracket_state['Final']
    if final_results.get(0):
        champion = final_results[0]['advancing_team']
        st.markdown('---')
        st.markdown(
            f"""<div style="background:linear-gradient(135deg,#f1c40f,#f39c12);border-radius:12px;
            padding:24px;text-align:center;color:#3a2c00">
            <div style="font-size:14px;font-weight:600">🏆 YOUR CHAMPION</div>
            <div style="font-size:32px;font-weight:800;margin-top:4px">{champion}</div>
            </div>""",
            unsafe_allow_html=True
        )

# ============================================
# PAGE 6 — KNOCKOUT ADVANCEMENT PROBABILITIES
# ============================================

elif page == 'Advancement Probabilities (Knockout)':
    st.title('Knockout Advancement Probabilities')
    st.markdown('Based on Monte Carlo simulations of the full knockout bracket. Available once the real bracket is finalized and the knockout Monte Carlo has been run.')

    try:
        knockout_prob_df = pd.read_csv('knockout_advancement_probabilities.csv')
        knockout_prob_df = knockout_prob_df.sort_values('champion%', ascending=False)

        st.dataframe(knockout_prob_df.reset_index(drop=True), use_container_width=True)

        st.markdown('---')
        st.subheader('Championship odds')

        fig, ax = plt.subplots(figsize=(10, max(4, len(knockout_prob_df) * 0.3)))
        top_teams = knockout_prob_df.head(16).sort_values('champion%', ascending=True)
        ax.barh(top_teams['team'], top_teams['champion%'], color='#1a6b3c')
        ax.set_xlabel('Championship probability (%)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except FileNotFoundError:
        st.info('`knockout_advancement_probabilities.csv` not found yet. Run the knockout Monte Carlo simulation in `main.py` once the Round of 32 bracket is set, then this page will populate automatically.')