import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import poisson
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style='darkgrid', palette='muted')

# ============================================
# 1. LOAD DATA
# ============================================

df = pd.read_csv('cleaned_data.csv')
df['date'] = pd.to_datetime(df['date'])
elo_df = pd.read_csv('elo_ratings.csv')
elo_ratings = dict(zip(elo_df['team'], elo_df['elo']))
rankings_df = pd.read_csv('fifa_rankings.csv')
prob_df = pd.read_csv('advancement_probabilities.csv')
fixtures_df = pd.read_csv('monte_carlo_fixtures.csv')

# ===========================================
# 2. COMMENTED OUT: Formation of Data
#    cleaned_data.csv AND elo_ratings.csv
#    Uncomment and run if raw data changes
# ===========================================

# df = pd.read_csv('results.csv')
# shootouts = pd.read_csv('shootouts.csv')
# shootouts['date'] = pd.to_datetime(shootouts['date'])
# df['date'] = pd.to_datetime(df['date'])

# df = df.merge(shootouts[['date', 'home_team', 'away_team', 'winner']],
#               on=['date', 'home_team', 'away_team'], how='left')

# tournament_weights = {
#     'FIFA World Cup': 4,
#     'UEFA Euro': 3,
#     'Copa América': 3,
#     'AFC Asian Cup': 3,
#     'Africa Cup of Nations': 3,
#     'UEFA Nations League': 2,
#     'CONMEBOL-UEFA Cup of Champions': 2,
#     'Friendly': 1
# }

# def get_result(row):
#     if pd.notna(row.get('winner')):
#         return 'H' if row['winner'] == row['home_team'] else 'A'
#     if row['home_score'] > row['away_score']:
#         return 'H'
#     elif row['home_score'] < row['away_score']:
#         return 'A'
#     else:
#         return 'D'

# df['result'] = df.apply(get_result, axis=1)
# df = df[df['date'].dt.year >= 2016]
# df = df.reset_index(drop=True)
# df = df.dropna(subset=['home_score', 'away_score'])
# df = df.reset_index(drop=True)

# elo_ratings = {}

# def get_elo(team):
#     return elo_ratings.get(team, 1000)

# def update_elo(home_team, away_team, home_score, away_score, tournament):
#     weight = tournament_weights.get(tournament, 2)
#     K = 8 * weight
#     home_elo = get_elo(home_team)
#     away_elo = get_elo(away_team)
#     expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
#     expected_away = 1 - expected_home
#     if home_score > away_score:
#         actual_home, actual_away = 1, 0
#     elif home_score < away_score:
#         actual_home, actual_away = 0, 1
#     else:
#         actual_home, actual_away = 0.5, 0.5
#     elo_ratings[home_team] = home_elo + K * (actual_home - expected_home)
#     elo_ratings[away_team] = away_elo + K * (actual_away - expected_away)
#     return elo_ratings[home_team], elo_ratings[away_team]

# home_elos, away_elos = [], []
# for _, row in df.iterrows():
#     home_elos.append(get_elo(row['home_team']))
#     away_elos.append(get_elo(row['away_team']))
#     update_elo(row['home_team'], row['away_team'],
#                row['home_score'], row['away_score'], row['tournament'])

# elo_df = pd.DataFrame([{'team': t, 'elo': r} for t, r in elo_ratings.items()])
# elo_df.to_csv('elo_ratings.csv', index=False)

# df['home_form'] = df.apply(lambda row: get_recent_form(row['home_team'], row['date']), axis=1)
# df['away_form'] = df.apply(lambda row: get_recent_form(row['away_team'], row['date']), axis=1)
# df['home_elo'] = home_elos
# df['away_elo'] = away_elos
# df['elo_difference'] = (df['home_elo'] - df['away_elo']).round(2)
# df['goal_difference'] = df['home_score'] - df['away_score']
# df['tournament_weight'] = df['tournament'].map(tournament_weights).fillna(2)
# df['home_ranking'] = df['home_team'].map(rankings_df.set_index('team')['points'])
# df['away_ranking'] = df['away_team'].map(rankings_df.set_index('team')['points'])
# df['ranking_difference'] = df['home_ranking'] - df['away_ranking']
# df.to_csv('cleaned_data.csv', index=False)

# ============================================
# 3. FEATURE ENGINEERING HELPERS
# ============================================

def get_recent_form(team, date, n=5):
    past = df[(df['date'] < date) &
              ((df['home_team'] == team) | (df['away_team'] == team))].tail(n)
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

avg_home_scored = df['home_score'].mean()
avg_away_scored = df['away_score'].mean()

# Precompute competitive matches once — reused in every predict_score_and_result call
competitive = df[df['tournament_weight'] >= 2].copy()
competitive_home_scores = competitive['home_score'].values
competitive_away_scores = competitive['away_score'].values
competitive_elo_diff = competitive['elo_difference'].values
competitive_results = competitive['result'].values
competitive_home_rankings = competitive['home_ranking'].values
competitive_away_rankings = competitive['away_ranking'].values
competitive_ranking_diff = competitive['ranking_difference'].values

# ============================================
# 4. TRAIN RANDOM FOREST MODEL
# ============================================

features = ['elo_difference', 'home_elo', 'away_elo', 'neutral', 'tournament_weight', 'home_form', 'away_form', 'home_ranking', 'away_ranking', 'ranking_difference']
target = 'result'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# # Uncomment to evaluate model performance
# y_pred = model.predict(X_test)
# print(f'Accuracy: {accuracy_score(y_test, y_pred):.2%}')
# print(classification_report(y_test, y_pred))

# ============================================
# 5. SCORE PREDICTOR
# ============================================

def predict_score_and_result(home_team, away_team):
    home_elo = elo_ratings.get(home_team, 1000)
    away_elo = elo_ratings.get(away_team, 1000)
    elo_diff = home_elo - away_elo
    home_ranking = rankings_df.set_index('team')['points'].get(home_team, 0)
    away_ranking = rankings_df.set_index('team')['points'].get(away_team, 0)
    ranking_diff = home_ranking - away_ranking

    # Vectorized weight calculation — no row by row apply
    elo_similarity = 1 / (1 + np.abs(competitive_elo_diff - elo_diff) / 100)
    ranking_similarity = 1 / (1 + np.abs(competitive_ranking_diff - ranking_diff) / 100)

    elo_direction_boost = np.ones(len(competitive))
    if elo_diff > 50:
        elo_direction_boost[competitive_results == 'H'] = 3.0
    elif elo_diff < -50:
        elo_direction_boost[competitive_results == 'A'] = 3.0
    else:
        elo_direction_boost[competitive_results == 'D'] = 3.0
    elo_direction_boost[competitive_results == 'D'] = np.maximum(
        elo_direction_boost[competitive_results == 'D'], 1.5
    )

    ranking_direction_boost = np.ones(len(competitive))
    if ranking_diff > 100:
        ranking_direction_boost[competitive_results == 'H'] = 3.0
    elif ranking_diff < -100:
        ranking_direction_boost[competitive_results == 'A'] = 3.0
    else:
        ranking_direction_boost[competitive_results == 'D'] = 3.0
    ranking_direction_boost[competitive_results == 'D'] = np.maximum(
        ranking_direction_boost[competitive_results == 'D'], 1.5
    )

    weights = elo_similarity * ranking_similarity * elo_direction_boost * ranking_direction_boost
    weights /= weights.sum()

    avg_home = np.dot(competitive_home_scores, weights)
    avg_away = np.dot(competitive_away_scores, weights)
    # Use weighted average as lambda for Poisson sampling
    avg_home += np.random.normal(0, 0.3)
    avg_away += np.random.normal(0, 0.3)

    # Sample from Poisson using weighted avg as expected goals
    h = int(np.random.poisson(max(0.1, avg_home)))
    a = int(np.random.poisson(max(0.1, avg_away)))

    if h > a:
        result = 'H'
    elif a > h:
        result = 'A'
    else:
        result = 'D'

    return h, a, result

# # Uncomment to evaluate Poisson accuracy
# def test_poisson_independent(row):
#     h, a, predicted = predict_score_and_result(row['home_team'], row['away_team'])
#     return predicted

# df['poisson_pred'] = df.apply(test_poisson_independent, axis=1)
# print(f'Poisson Accuracy: {accuracy_score(df["result"], df["poisson_pred"]):.2%}')
# print(classification_report(df['result'], df['poisson_pred']))

# ============================================
# 6. BUILD 2026 WORLD CUP FIXTURES
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

# fixtures = []
# for group, teams in groups.items():
#     for home, away in combinations(teams, 2):
#         fixtures.append({'group': group, 'home_team': home, 'away_team': away})

# fixtures_df = pd.DataFrame(fixtures)

# fixtures_df['home_elo'] = fixtures_df['home_team'].map(elo_ratings)
# fixtures_df['away_elo'] = fixtures_df['away_team'].map(elo_ratings)
# fixtures_df['elo_difference'] = (fixtures_df['home_elo'] - fixtures_df['away_elo']).round(2)
# fixtures_df['neutral'] = True
# fixtures_df['tournament_weight'] = 4

# fixtures_df['home_form'] = fixtures_df['home_team'].apply(
#     lambda team: get_recent_form(team, pd.Timestamp('2026-06-11'))
# )
# fixtures_df['away_form'] = fixtures_df['away_team'].apply(
#     lambda team: get_recent_form(team, pd.Timestamp('2026-06-11'))
# )

# fixtures_df['home_ranking'] = fixtures_df['home_team'].map(rankings_df.set_index('team')['points'])
# fixtures_df['away_ranking'] = fixtures_df['away_team'].map(rankings_df.set_index('team')['points'])
# fixtures_df['ranking_difference'] = fixtures_df['home_ranking'] - fixtures_df['away_ranking']

# ============================================
# 7. PREDICT MATCH OUTCOMES + SCORES
# ============================================

# def predict_final(row):
#     feature_vals = pd.DataFrame([row[features]])
#     proba = model.predict_proba(feature_vals)[0]
#     classes = model.classes_
#     proba_dict = dict(zip(classes, proba))
    
#     max_prob = max(proba_dict.values())
#     predicted = max(proba_dict, key=proba_dict.get)
    
#     # allow upset based on remaining probability
#     upset_chance = (1 - max_prob) ** 1.85
#     if np.random.random() < upset_chance:
#         # Sample from non-dominant outcomes weighted by their probabilities
#         other_classes = [c for c in classes if c != predicted]
#         other_probs = np.array([proba_dict[c] for c in other_classes])
#         other_probs /= other_probs.sum()
#         predicted = np.random.choice(other_classes, p=other_probs)

#     h, a, sampled_result = predict_score_and_result(row['home_team'], row['away_team'])
#     attempts = 0
#     while sampled_result != predicted and attempts < 100:
#         h, a, sampled_result = predict_score_and_result(row['home_team'], row['away_team'])
#         attempts += 1

#     if sampled_result != predicted:
#         if predicted == 'H': return 1, 0, 'H'
#         if predicted == 'A': return 0, 1, 'A'
#         return 1, 1, 'D'

#     return h, a, predicted

# fixtures_df[['home_goals', 'away_goals', 'predicted_result']] = fixtures_df.apply(
#     predict_final, axis=1, result_type='expand'
# )

# # Uncomment to test accuracy on historical data
# df['final_pred'] = df.apply(lambda row: predict_final(row)[2], axis=1)
# print(f'Final Accuracy: {accuracy_score(df["result"], df["final_pred"]):.2%}')
# print(classification_report(df['result'], df['final_pred']))

# ============================================
# 8. SIMULATE GROUP STAGE STANDINGS
# ============================================

def simulate_group(group_name, predictions_df):
    group_matches = predictions_df[predictions_df['group'] == group_name]
    teams = groups[group_name]

    standings = {team: {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0} for team in teams}

    for _, match in group_matches.iterrows():
        home = match['home_team']
        away = match['away_team']
        home_goals = match['home_goals']
        away_goals = match['away_goals']
        result = match['predicted_result']

        standings[home]['P'] += 1
        standings[away]['P'] += 1
        standings[home]['GF'] += home_goals
        standings[home]['GA'] += away_goals
        standings[away]['GF'] += away_goals
        standings[away]['GA'] += home_goals

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

# # Uncomment to verify all match scores
# print(fixtures_df[['group', 'home_team', 'away_team', 'home_goals', 'away_goals', 'predicted_result']].to_string())

# for group_name in sorted(groups.keys()):
#     print(f'\n=== GROUP {group_name} ===')
#     print(simulate_group(group_name, fixtures_df).to_string())


# ============================================
# MONTE CARLO SIMULATION (run once, then comment out)
# ============================================

# from collections import Counter

# N_SIMULATIONS = 1000

# match_results = {i: [] for i in range(len(fixtures_df))}
# match_scores = {i: [] for i in range(len(fixtures_df))}

# # Track position counts per team
# all_teams = [team for group in groups.values() for team in group]
# position_counts = {team: {'1st': 0, '2nd': 0, '3rd_qual': 0, '3rd_elim': 0, '4th': 0} for team in all_teams}

# print('Running Monte Carlo simulation...')

# for sim in range(N_SIMULATIONS):
#     if sim % 50 == 0:
#         print(f'Simulation {sim}/{N_SIMULATIONS}...')
    
#     sim_fixtures = fixtures_df.copy()
#     sim_fixtures[['home_goals', 'away_goals', 'predicted_result']] = sim_fixtures.apply(
#         predict_final, axis=1, result_type='expand'
#     )
    
#     for i, row in sim_fixtures.iterrows():
#         match_results[i].append(row['predicted_result'])
#         match_scores[i].append((int(row['home_goals']), int(row['away_goals'])))

#     # Simulate all groups and collect third place teams
#     third_place_teams = []

#     for group_name in sorted(groups.keys()):
#         standings = simulate_group(group_name, sim_fixtures)
#         teams_in_order = standings.index.tolist()

#         # Record 1st, 2nd, 4th directly
#         position_counts[teams_in_order[0]]['1st'] += 1
#         position_counts[teams_in_order[1]]['2nd'] += 1
#         position_counts[teams_in_order[3]]['4th'] += 1

#         # Collect 3rd place team with their stats for cross-group ranking
#         third_team = teams_in_order[2]
#         third_place_teams.append({
#             'team': third_team,
#             'Pts': standings.loc[third_team, 'Pts'],
#             'GD': standings.loc[third_team, 'GD'],
#             'GF': standings.loc[third_team, 'GF']
#         })

#     # Rank all 12 third place teams by Pts → GD → GF
#     third_df = pd.DataFrame(third_place_teams)
#     third_df = third_df.sort_values(['Pts', 'GD', 'GF'], ascending=False).reset_index(drop=True)

#     # Top 8 qualify, bottom 4 eliminated
#     for i_third, row in third_df.iterrows():
#         if i_third < 8:
#             position_counts[row['team']]['3rd_qual'] += 1
#         else:
#             position_counts[row['team']]['3rd_elim'] += 1

# print('Done! Processing results...')

# # Take most common result and score per match
# for i in range(len(fixtures_df)):
#     most_common_result = Counter(match_results[i]).most_common(1)[0][0]
#     matching_scores = [s for s, r in zip(match_scores[i], match_results[i]) if r == most_common_result]
#     most_common_score = Counter(matching_scores).most_common(1)[0][0]
#     fixtures_df.at[i, 'predicted_result'] = most_common_result
#     fixtures_df.at[i, 'home_goals'] = most_common_score[0]
#     fixtures_df.at[i, 'away_goals'] = most_common_score[1]

# fixtures_df.to_csv('monte_carlo_fixtures.csv', index=False)

# # Build and save probability table
# prob_rows = []
# for group_name, teams in groups.items():
#     for team in teams:
#         counts = position_counts[team]
#         prob_rows.append({
#             'team': team,
#             'group': group_name,
#             '1st%': round(counts['1st'] / N_SIMULATIONS * 100, 1),
#             '2nd%': round(counts['2nd'] / N_SIMULATIONS * 100, 1),
#             '3rd_qual%': round(counts['3rd_qual'] / N_SIMULATIONS * 100, 1),
#             '3rd_elim%': round(counts['3rd_elim'] / N_SIMULATIONS * 100, 1),
#             '4th%': round(counts['4th'] / N_SIMULATIONS * 100, 1),
#             'advancement%': round((counts['1st'] + counts['2nd'] + counts['3rd_qual']) / N_SIMULATIONS * 100, 1)
#         })

# prob_df = pd.DataFrame(prob_rows)
# prob_df.to_csv('advancement_probabilities.csv', index=False)

# print('\nAdvancement Probabilities:')
# print(prob_df.to_string(index=False))
# print('\nSaved to advancement_probabilities.csv!')

# ============================================
# 9. VISUALIZATIONS (coming next)
# ============================================

# ============================================
# VISUAL 1: Match Results (Monte Carlo Fixtures)
# ============================================

fig, axes = plt.subplots(6, 2, figsize=(22, 60))
fig.suptitle('2026 FIFA World Cup — Predicted Match Results', 
             fontsize=20, fontweight='bold', y=0.995)

group_list = sorted(groups.keys())

for idx, group_name in enumerate(group_list):
    ax = axes[idx // 2][idx % 2]
    group_matches = fixtures_df[fixtures_df['group'] == group_name].copy()
    group_matches = group_matches.reset_index(drop=True)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(group_matches) - 0.5)
    ax.axis('off')
    ax.set_title(f'Group {group_name}', fontsize=14, fontweight='bold', pad=10)

    # Column headers
    ax.text(0.02, len(group_matches) - 0.1, 'Home', fontsize=12,
            fontweight='bold', color='#333333', va='center')
    ax.text(0.5, len(group_matches) - 0.1, 'Score', fontsize=12,
            fontweight='bold', color='#333333', va='center', ha='center')
    ax.text(0.98, len(group_matches) - 0.1, 'Away', fontsize=12,
            fontweight='bold', color='#333333', va='center', ha='right')

    ax.axhline(y=len(group_matches) - 0.3, color='#cccccc', linewidth=0.8)

    for i, match in group_matches.iterrows():
        y = len(group_matches) - 1 - i
        result = match['predicted_result']
        home_goals = int(match['home_goals'])
        away_goals = int(match['away_goals'])

        # Color winners bold
        home_weight = 'bold' if result == 'H' else 'normal'
        away_weight = 'bold' if result == 'A' else 'normal'
        home_color = '#1a6b3c' if result == 'H' else ('#c0392b' if result == 'A' else '#666666')
        away_color = '#1a6b3c' if result == 'A' else ('#c0392b' if result == 'H' else '#666666')

        ax.text(0.02, y, match['home_team'], fontsize=11,
                fontweight=home_weight, color=home_color, va='center')
        ax.text(0.5, y, f'{home_goals} — {away_goals}', fontsize=10,
                fontweight='bold', color='#222222', va='center', ha='center')
        ax.text(0.98, y, match['away_team'], fontsize=11,
                fontweight=away_weight, color=away_color, va='center', ha='right')

        if i < len(group_matches) - 1:
            ax.axhline(y=y - 0.4, color='#eeeeee', linewidth=0.5)

# Legend
win_patch = mpatches.Patch(color='#1a6b3c', label='Winner')
loss_patch = mpatches.Patch(color='#c0392b', label='Loser')
draw_patch = mpatches.Patch(color='#666666', label='Draw')
fig.legend(handles=[win_patch, loss_patch, draw_patch],
           loc='lower center', ncol=3, fontsize=11, 
           bbox_to_anchor=(0.5, 0.001))

plt.tight_layout(rect=[0, 0.01, 1, 0.995])
plt.savefig('match_results.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved match_results.png')

# ============================================
# VISUAL 2: Advancement Probabilities by Group
# ============================================

fig, axes = plt.subplots(6, 2, figsize=(22, 60))
fig.suptitle('2026 FIFA World Cup — Advancement Probabilities by Group',
             fontsize=20, fontweight='bold', y=0.995)

colors = {
    '1st%': '#1a6b3c',
    '2nd%': "#16b71b",
    '3rd_qual%': "#28f508",
    '3rd_elim%': "#e30909",
    '4th%': "#ff0000"
}

labels = {
    '1st%': '1st place',
    '2nd%': '2nd place',
    '3rd_qual%': '3rd (qualify)',
    '3rd_elim%': '3rd (eliminated)',
    '4th%': '4th place'
}

for idx, group_name in enumerate(group_list):
    ax = axes[idx // 2][idx % 2]
    group_data = prob_df[prob_df['group'] == group_name].copy()
    group_data = group_data.sort_values('advancement%', ascending=True)

    teams = group_data['team'].tolist()
    cols = ['4th%', '3rd_elim%', '3rd_qual%', '2nd%', '1st%']
    left = np.zeros(len(teams))

    for col in cols:
        values = group_data[col].values
        bars = ax.barh(teams, values, left=left,
                      color=colors[col], label=labels[col], height=0.6)
        
        # Add percentage labels inside bars if wide enough
        for bar, val in zip(bars, values):
            if val >= 8:
                ax.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_y() + bar.get_height() / 2,
                       f'{val:.0f}%', ha='center', va='center',
                       fontsize=8, color='white', fontweight='bold')
        left += values

    ax.set_xlim(0, 100)
    ax.set_title(f'Group {group_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Probability (%)', fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=8)
    ax.axvline(x=50, color='white', linewidth=0.8, linestyle='--', alpha=0.5)

# Single shared legend
handles = [mpatches.Patch(color=colors[c], label=labels[c]) for c in cols[::-1]]
fig.legend(handles=handles, loc='lower center', ncol=5,
           fontsize=11, bbox_to_anchor=(0.5, 0.001))

plt.tight_layout(rect=[0, 0.01, 1, 0.995])
plt.savefig('advancement_probabilities.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved advancement_probabilities.png')