# final_project_ML

## Question and Background

In this project, we study whether the characteristics of Minecraft mobs can be used to predict whether a mob is Hostile or Peaceful. This is a classification problem because the goal is to assign each mob to one of two categories based on its attributes. The central question is: Can features such as health, damage, spawning behavior, reproduction ability, and debut year help classify Minecraft mobs by behavior type?

This question is interesting because Minecraft mobs are designed with distinct gameplay roles, and those roles are reflected in their traits. Hostile mobs are generally intended to attack or threaten the player, while peaceful mobs are usually passive or neutral in gameplay. Because these roles are tied to measurable or categorical features, this dataset offers a good opportunity to test whether machine learning models can learn those patterns. In other words, the project explores whether the design logic of the game can be captured through data.

The dataset comes from Kaggle and includes Minecraft-related data for blocks, items, mobs, biomes, and other game elements. For this project, only the Mobs.csv file was used. The mob dataset contains information such as mob name, health points, maximum damage, spawn behavior, reproductive requirements, and debut date. Since the response variable is the mob’s behavior type, this dataset is well suited for supervised classification. It also contains a mix of numerical and categorical variables, making it a strong example for using preprocessing pipelines and comparing multiple tree-based models.

## Exploratory Data Analysis

The first step in the project was exploring the mob dataset to understand its structure and identify which variables might matter most for classification. Summary statistics and .info() were used to inspect data types, missing values, and the overall size of the dataset. Since the dataset is relatively small, understanding each variable carefully was especially important before modeling.

One of the first observations was the distribution of the target variable, behaviorTypes. A bar chart showed the counts of different mob behavior categories. The dataset included a Conditional category, representing mobs whose hostility depends on player behavior or context. Because these mobs behave more similarly to threatening mobs than fully passive ones, this category was recoded into Hostile. This simplified the problem into a cleaner binary classification task: Hostile vs Peaceful.

The exploratory analysis also revealed that some columns were not useful in their original form. For example, name and ID acted more like identifiers than predictive features, so they were removed. Some variables also needed feature engineering. Missing values in maxDamage were filled with 0, since a missing damage value often implies that the mob does not deal damage. The reproductiveRequirement field was transformed into a binary variable indicating whether the mob can reproduce, which made it more useful for the model. In addition, debutDate was converted into a new numerical variable, debutYear, so that time-related information could be included in the model.

Overall, the EDA showed that the most promising predictors were those tied to gameplay design, such as damage, spawning behavior, and reproduction. These findings shaped the preprocessing steps and gave a clear story for the analysis: mob behavior appears to be connected to other in-game characteristics, and those connections may allow a model to classify mobs accurately.

## Methods

To answer the project question, we built machine learning models that classify mobs as Hostile or Peaceful based on their features. The target variable was behaviorTypes, and the remaining columns after cleaning and feature engineering were used as predictors. The dataset was split into training, validation, and testing sets so that model development and final evaluation could be kept separate.

A pipeline-based approach was used for model development. This is an important part of a robust data science workflow because it ensures that preprocessing steps are applied consistently across training and test data. Categorical variables were encoded using OneHotEncoder, while the model itself was placed inside the same pipeline. This reduced the risk of data leakage and made the workflow easier to reproduce.

The first model was a Decision Tree Classifier. This model was a strong starting point because decision trees are interpretable and can naturally handle nonlinear relationships. The initial decision tree performed very well on the training data, with a training accuracy of about 97.8%, but its test accuracy was lower at 81.25%. This suggested that the model may have been slightly overfitting the training data.

To improve the decision tree, GridSearchCV was used to tune hyperparameters such as maximum depth, split criterion, minimum samples split, and minimum samples leaf. This tuning process identified the best tree as one with a maximum depth of 3 using the gini criterion. The tuned model achieved a cross-validation accuracy of 93.33%, but the test accuracy remained 81.25%, meaning the improvements in validation performance did not substantially improve test-set performance.

Finally, a Random Forest Classifier was built as a more advanced model. Random forests combine many decision trees and typically improve generalization by reducing overfitting. This model outperformed the single-tree models, achieving a test accuracy of 93.75%. Because it aggregates decisions across many trees, the random forest was able to capture more stable patterns in the data and produced the strongest overall classification performance in the project.

## Evaluation of the Model

Model evaluation focused on classification performance on the held-out test set. Since this is a binary classification problem, accuracy alone is not enough, so additional metrics such as precision, recall, F1-score, confusion matrices, and ROC analysis were used.

The original and tuned decision tree models both achieved a test accuracy of 81.25%. Their classification reports showed reasonably balanced performance, but they still made several mistakes. For the decision tree, the model had stronger recall for Hostile mobs than for Peaceful mobs, suggesting that it was slightly better at identifying threatening mobs than passive ones. The confusion matrix showed that the model made 3 mistakes out of 16 test cases, which is solid but leaves room for improvement.

The random forest performed best by a clear margin. It achieved 93.75% test accuracy, with a precision of 0.90 and recall of 1.00 for Hostile mobs, and a precision of 1.00 and recall of 0.86 for Peaceful mobs. These results indicate that the random forest was especially strong at detecting Hostile mobs while still performing very well on Peaceful mobs. In the test set, it only misclassified one mob, a pufferfish, which was predicted as Hostile instead of Peaceful. This error is interesting because the pufferfish can damage players in certain situations, so its traits may overlap with more hostile mobs.

Cross-validation results also helped compare the models. The decision tree showed a mean cross-validation accuracy of 93.33%, while the random forest had a mean cross-validation accuracy of about 84.89%, though with more variability. Even though the decision tree had slightly stronger average cross-validation accuracy, the random forest generalized better to the test set. This highlights why the final test evaluation is so important: a model that looks strong during training or tuning does not always perform best on unseen data.

Overall, the evaluation suggests that the Random Forest is the strongest final model for this project because it produced the highest test accuracy and the most balanced classification performance.

## Conclusions

This project shows that Minecraft mob characteristics can be used effectively to predict whether a mob is Hostile or Peaceful. The models were able to learn meaningful patterns from gameplay-related features such as damage, spawn behavior, reproduction, and debut year. Among the models tested, the Random Forest Classifier performed best, reaching 93.75% accuracy on the test set and misclassifying only one mob.

These results suggest that mob behavior in Minecraft is not random but is strongly tied to other design features in the game. For example, mobs with higher damage values or certain spawning patterns are more likely to be hostile, while peaceful mobs tend to share different trait combinations. In this way, the model captures some of the game’s internal design logic through data.

At the same time, there are important limitations. The dataset is fairly small, so the results may not be as stable as they would be with a larger sample. Some variables were simplified during preprocessing, such as collapsing Conditional mobs into Hostile and converting reproduction into a binary feature. Those choices made modeling easier, but they may also remove some nuance from the original data. In addition, the dataset reflects the way the game was structured at one point in time, so future Minecraft updates could change mob behavior and affect model relevance.

Despite these limitations, the project successfully demonstrates the full data science lifecycle: asking a question, exploring the data, preparing features, building models, evaluating results, and drawing conclusions. The final outcome is that machine learning can classify Minecraft mobs quite accurately, and the random forest provides the best overall solution for this dataset.

## Team Contribution

Our team divided responsibilities across the stages of the data science lifecycle while still collaborating on major decisions throughout the project.

Data Engineer: (Michael) Responsible for setting up the notebook environment, importing the dataset, cleaning the data, handling missing values, and preparing features for modeling.
Researcher: (Michael & Braden) Developed the project question, explained the background of the dataset, and connected the analysis to the logic of Minecraft mob behavior.
Model Builder: (Braden) Built the decision tree and random forest pipelines, ran training and hyperparameter tuning, and compared model performance.
Red Team / Evaluator: (Mani) Reviewed model outputs, checked for overfitting, interpreted confusion matrices and classification reports, and analyzed the misclassified examples.
Final Reviewer: (Braden) Organized the notebook, improved clarity in markdown explanations, and ensured the final project told a clear and consistent story.

Although different team members may have focused on different areas, the final notebook reflects collaboration across data preparation, modeling, interpretation, and presentation.
