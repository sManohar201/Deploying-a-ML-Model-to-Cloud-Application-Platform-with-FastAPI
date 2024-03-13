# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is a classifier to predict whether a person makes over 50k a year.

    - Model: Random forest classifier

## Intended Use

The model can be used to predict the individual's income levels as persented in the Census features. The potential stakeholders of this model are

    - Data engineers
    - Machine learning engineers
    - executive team
    - large organizations
    - marketing team

## Training Data

The training data is the publicly available Census Bureau data from UCI library. The data consists of 32561 samples and 15 features. The target label "salary", has two categories ('<=50K', '>50K'). 

## Evaluation Data

The validation data is extracted from the training data, with a 80:20 ratio split. 

## Metrics

Testing metris:
    - Precision: 0.74 (ratio of true positives to the total predicted positives)
    - Recall: 0.64 (ratio of true positives to the total actual positives)
    - F-beta: 0.68 (harmonic weight of precision and recall)

## Ethical Considerations

    - The dataset employed for model training and evaluation has been rigorously anonymized to safeguard privacy and prevent the inclusion of personally identifiable information (PII).
    - The model development process prioritized fairness principles. Measures were taken to mitigate potential biases and ensure that the model does not exhibit discriminatory behavior across different demographic groups. 
    - Analysis of model results reveals discernible differences in outcomes based on the 'race' feature. Further investigation is warranted to understand the root causes of these disparities and explore potential mitigation strategies.

## Caveats and Recommendations

This dataset was extracted from an outdated Census database. Due to its age, it may not accurately reflect the current population's demographics and socioeconomic realities. However, it remains valuable for training machine learning classification models and exploring related problems.
