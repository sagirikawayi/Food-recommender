📝 Update Log (Changelog)
[v1.1.0] - 2026-04-03
"The Precision & Standards Update"

🚀 New Features
Precise Nutrient Control: Added Number Input fields synchronized with Sliders for exact calorie and protein targeting.

EU Nutritional Standards Guidance: Integrated real-time feedback cards based on EU Regulation (EC) No 1924/2006 to categorize food:

Energy: Low Energy (≤120kcal), Standard, High Energy.

Protein: Low Protein, Source of Protein (≥5g), High Protein (≥12g).

Dual-Binding UI: Implemented st.session_state and callback functions to ensure seamless synchronization between sliders and manual input boxes.

🛠️ Technical Improvements
Language Localization: Migrated the entire User Interface (UI) to English for international professional standards.

Refactored Logic: Optimized the KNN feature vector scaling to align with the new precise input system.

Project Restructuring: Renamed the entry point from test.py to Food.py for better repository clarity.

🐞 Bug Fixes
Fixed the DuplicateWidgetID error caused by shared keys between input components.

Corrected the Euclidean distance calculation to handle edge cases with zero-variance inputs.

[v1.0.0] - 2026-03-25
"Initial Release: AI Food Recommender"

Base KNN algorithm implementation using scikit-learn.

Initial UI with Streamlit.

Data integration for healthy_foods_database.csv.

# 📝 Update Log (Changelog)

## [v2.0.0] - 2026-04-05
*"The Global Standards & Algorithm Transparency Update"*

---

### 🚀 New Features

* **International & Domestic Standards Integration**: Fully migrated the evaluation logic to professional regulatory frameworks:
    * **UK FSA Traffic Light System**: Implemented risk-based warnings for **Fat, Sugar, and Sodium** using Green (Low), Amber (Medium), and Red (High) visual indicators.
    * **China GB 28050-2011**: Integrated the national standard for **High Protein** claims (≥ 12g/100g) and **Source of Protein** (≥ 6g/100g) certification logic.

* **Algorithm X-Ray Vision**: Added a new UI toggle to reveal the underlying AI decision-making process. Users can now choose between:
    * **User Mode**: Clean progress bars showing Match Confidence.
    * **Dev Mode**: Raw **Euclidean Distance** and precise confidence percentages.

* **Dynamic 6D Search Engine**: Expanded the search space to 6 nutritional dimensions with a "Feature Masking" system. The KNN model now dynamically re-weights its vector space based on which nutrients the user chooses to "Activate."

---

### 🛠️ Technical Improvements

* **State Logic Optimization**: Completely refactored the `sync_val` callback system. Decoupled `value` parameters from key-based widgets to resolve Streamlit's initialization warnings and improve input reactivity.

* **Scaling Architecture**: Enhanced the `MinMaxScaler` pipeline to handle the massive magnitude gap between Sodium (mg) and other macronutrients (g), ensuring balanced feature influence in the KNN calculation.

* **Responsive Result Grid**: Refactored the results container to use dynamic column mapping, ensuring the UI remains clean regardless of how many nutrient metrics are being displayed.

---

### 🐞 Bug Fixes

* **Fixed SessionState Conflict**: Eliminated the "The widget with key 'cal_input' was created with a default value..." error that occurred on app startup.

* **Restored Match Metrics**: Fixed a logic break in the results loop where the `match_score` was failing to render, restoring the visibility of match percentages and progress bars.

* **Toggle Responsiveness**: Corrected the conditional rendering block for `show_math` to ensure real-time UI updates without requiring a page refresh.
