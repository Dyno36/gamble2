import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import os

st.title("Advanced Player Prop Betting Simulator")
st.write("Simulate player prop outcomes with Monte Carlo and Bayesian updating, incorporating usage, matchups, and more!")

# File path for saving/loading data
DATA_FILE = "player_data.json"

# Load existing player data
def load_all_players():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            return json.load(file)
    return {}

# Save all players
def save_all_players(players):
    with open(DATA_FILE, "w") as file:
        json.dump(players, file, indent=4)

# Load players into dropdown
players = load_all_players()
player_names = list(players.keys())

# Sidebar player selection
st.sidebar.header("Select or Enter Player Data")
selected_player = st.sidebar.selectbox("Choose a Player", ["New Player"] + player_names)

if selected_player != "New Player":
    player_data = players[selected_player]
else:
    player_data = {}

# Sidebar inputs
player_name = st.sidebar.text_input("Player Name", player_data.get("player_name", "Example Player"))
player_position = st.sidebar.selectbox("Player Position", ["PG", "SG", "SF", "PF", "C"],
                                       index=["PG", "SG", "SF", "PF", "C"].index(player_data.get("player_position", "PG")))
mean_points = st.sidebar.number_input("Average Points per Game", value=player_data.get("mean_points", 20.0))
std_dev_points = st.sidebar.number_input("Standard Deviation (Points)", value=player_data.get("std_dev_points", 1.0))
games_played = st.sidebar.number_input("Number of Games Played", value=player_data.get("games_played", None))

# Recent performance
st.sidebar.header("Recent Performance")
recent_avg_points = st.sidebar.number_input("Recent Average Points", value=player_data.get("recent_avg_points", 22.0))
recent_games = st.sidebar.number_input("Number of Recent Games", value=player_data.get("recent_games", 5))

# Opponent defense
st.sidebar.header("Opponent Defense Stats")
opp_points_allowed_position = st.sidebar.number_input(f"Opponent Points Allowed to {player_position}",
                                                      value=player_data.get("opp_points_allowed_position", 22.0))

# Minutes adjustment
st.sidebar.header("Minutes Adjustment")
projected_minutes = st.sidebar.number_input("Projected Minutes", value=player_data.get("projected_minutes", 35.0))
avg_minutes = st.sidebar.number_input("Player Average Minutes", value=player_data.get("avg_minutes", 34.0))

# Floor adjustment
st.sidebar.header("Floor Adjustment")
floor_percentage = st.sidebar.slider("Set Floor Percentage", 0, 100, player_data.get("floor_percentage", 50))

# Bet details
st.sidebar.header("Bet Details")
line = st.sidebar.number_input("Sportsbook Line", value=player_data.get("line", 20.5))
odds = st.sidebar.number_input("Bet Odds (e.g., -110 for American odds)", value=player_data.get("odds", -110))
simulations = st.sidebar.slider("Number of Monte Carlo Simulations", 1000, 20000, player_data.get("simulations", 10000))

# Save or delete player data
if st.sidebar.button("Save Player Data"):
    players[player_name] = {
        "player_name": player_name,
        "player_position": player_position,
        "mean_points": mean_points,
        "std_dev_points": std_dev_points,
        "games_played": games_played,
        "recent_avg_points": recent_avg_points,
        "recent_games": recent_games,
        "opp_points_allowed_position": opp_points_allowed_position,
        "line": line,
        "odds": odds,
        "simulations": simulations,
        "floor_percentage": floor_percentage,
        "projected_minutes": projected_minutes,
        "avg_minutes": avg_minutes
    }
    save_all_players(players)
    st.sidebar.success(f"{player_name} data saved successfully!")

if st.sidebar.button("Delete Player Data") and selected_player != "New Player":
    players.pop(selected_player, None)
    save_all_players(players)
    st.sidebar.success(f"{selected_player} data deleted successfully!")

# Simulation functions
def bayesian_update(prior_mu, prior_sigma, recent_mu, recent_games):
    if games_played is None or games_played == 0:
        return prior_mu, prior_sigma
    posterior_mu = (prior_mu / prior_sigma**2 + recent_games * recent_mu / prior_sigma**2) / (1 / prior_sigma**2 + recent_games / prior_sigma**2)
    posterior_sigma = np.sqrt(1 / (1 / prior_sigma**2 + recent_games / prior_sigma**2))
    return posterior_mu, posterior_sigma

def monte_carlo_simulation(mu, sigma, sims):
    return np.random.normal(mu, sigma, sims)

def calculate_ev(prob_over, odds):
    odds_decimal = 1 + (100 / abs(odds)) if odds < 0 else (odds / 100) + 1
    return (prob_over * odds_decimal) - (1 - prob_over)

def apply_floor_adjustment(projected_points, opp_points_allowed, floor_percentage):
    floor_value = opp_points_allowed * (floor_percentage / 100)
    floor_triggered = projected_points < floor_value
    return max(projected_points, floor_value), floor_triggered

def calculate_projected_points(mean_points, recent_avg_points, opp_points_allowed, projected_minutes, avg_minutes):
    weighted_avg = (0.6 * mean_points) + (0.4 * recent_avg_points)
    opponent_adjustment = (opp_points_allowed - mean_points) * 0.25
    minutes_adjustment = (projected_minutes / avg_minutes) if avg_minutes > 0 else 1
    projected_points = (weighted_avg + opponent_adjustment) * minutes_adjustment
    return projected_points

def get_bet_recommendation(edge_percentage):
    if edge_percentage >= 50:
        return "Very Strong Over Bet"
    elif edge_percentage >= 31:
        return "Strong Over Bet"
    elif edge_percentage >= 10:
        return "Good Over Bet"
    elif edge_percentage <= -50:
        return "Very Strong Under Bet"
    elif edge_percentage <= -31:
        return "Strong Under Bet"
    elif edge_percentage <= -10:
        return "Good Under Bet"
    else:
        return "No Strong Bet Recommendation"

# Calculate projections
posterior_mu, posterior_sigma = bayesian_update(mean_points, std_dev_points, recent_avg_points, recent_games)
projected_points = calculate_projected_points(mean_points, recent_avg_points, opp_points_allowed_position, projected_minutes, avg_minutes)

# Run Monte Carlo simulation
simulated_points = monte_carlo_simulation(projected_points, posterior_sigma, simulations)

final_points, floor_triggered = apply_floor_adjustment(projected_points, opp_points_allowed_position, floor_percentage)

prob_over_line = np.mean(simulated_points > line)
ev = calculate_ev(prob_over_line, odds)
edge_percentage = (prob_over_line * 100) - 50
bet_recommendation = get_bet_recommendation(edge_percentage)

# Results
st.subheader(f"Results for {player_name} ({player_position})")
if floor_triggered:
    st.markdown(f"**Projected Points:** <span style='color: orange;'>{final_points:.2f} (Floor Applied)</span>", unsafe_allow_html=True)
else:
    st.write(f"**Projected Points:** {final_points:.2f}")

st.write(f"**Standard Deviation:** ±{posterior_sigma:.2f} points")
st.write(f"**Probability of Hitting Over {line} Points:** {prob_over_line * 100:.2f}%")
st.write(f"**Expected Value (EV):** {ev:.2f}")
st.write(f"**Edge vs Line (%):** {edge_percentage:.2f}%")
st.write(f"**Bet Recommendation:** {bet_recommendation}")

# Plot results
st.subheader("Simulated Outcome Distribution")
fig, ax = plt.subplots()
ax.hist(simulated_points, bins=30, color="green", alpha=0.6, edgecolor="white")
ax.axvline(line, color="red", linestyle="--", label=f"Sportsbook Line: {line}")
ax.axvline(final_points, color="yellow", linestyle="--", label=f"Projected Points: {final_points:.2f}")
ax.legend()

st.pyplot(fig)

# Explanation Section
st.subheader("Understanding the Metrics")

st.markdown("""
### 📊 **Projected Points:**  
This is the player’s expected point output based on historical data, recent performance, and opponent defense. If the projection falls below the floor (based on opponent points allowed and your set floor percentage), the floor value is applied.

### 📈 **Standard Deviation:**  
This shows the expected variability in the player’s scoring output. A higher standard deviation means more variability, while a lower one indicates more consistent performance. The percentage tells you how much this variability is relative to the projection.

### 🎯 **Probability of Hitting Over the Line:**  
The chance (in percentage) that the player will score more than the sportsbook line, based on simulated game outcomes.

### 💡 **Expected Value (EV):**  
EV tells you whether a bet is profitable in the long run.  
- **Positive EV:** The bet has value — you’d make money over time if you placed this bet repeatedly.  
- **Negative EV:** The bet loses value over time.  

For example, if a bet has an EV of **+0.15**, it means you can expect to make **$0.15 profit per $1 wagered** in the long run.

### 🔍 **Edge vs Line Percentage:**  
This shows the difference between the model’s projection and the sportsbook line:  
- **Positive %:** The model favors the **Over**.  
- **Negative %:** The model favors the **Under**.  
- **Range:** The edge percentage is capped at ±50%, since probabilities can’t exceed 100%.

### ✅ **Bet Recommendation:**  
Based on the edge percentage:  
- **10% to 30%:** Good Bet (Over)  
- **31% to 49%:** Strong Bet (Over)  
- **50%+:** Very Strong Bet (Over)  
- **-10% to -30%:** Good Bet (Under)  
- **-31% to -49%:** Strong Bet (Under)  
- **-50%+:** Very Strong Bet (Under)       

This helps you quickly see whether a bet is worth considering and how confident the model is.
""")