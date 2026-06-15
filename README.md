FIFA 26
A project made to simulate the 72 group stage matches of the 2026 FIFA World Cup

<img width="1918" height="928" alt="Screenshot 2026-06-14 131019" src="https://github.com/user-attachments/assets/ebbbddb9-f4bb-42d8-be37-f6ab06625e4e" />

Try it out yourself at: https://fifa-26.streamlit.app/

Features
* Simulate each match yourself on the Match Simulator Screen
* View consensus result, and actual result once it occurs, on the Match Simulator Screen
* View simulated group standings, another consensus result through numerous trials, on the Group Standing Screen
* View advancement probabilities of each team, by position within their group on the Advancement Probabilities Section
* A quick overview of all consensus match results on the All Match Results Screen

How it works
A Random Forest model was used to choose the outcome of the match as it had the most variety and utilized all metrics to its advantage
A Poisson distribution model was used only for scoring purposes, as it leaned heavily toward selecting draws

Credits
Claude was used to help guide this project's structure
martj42's kaggle dataset titled "International football results from 1872 to 2026" was the dataset used to train models with
