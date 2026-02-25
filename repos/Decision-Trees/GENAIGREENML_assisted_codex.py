import csv
import random
import math
import pandas as pd


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATASET_HEADERS = "town,apartments_condos_multis_per_residential_parcels_2011,assessed_home_value_changes_2009-2013,births_per_1000_residents_2010,boaters_per_10000_residents_2012,burglaries_per_10000_residents_2011,cars_motorcycles_&_trucks_average_age_2012,cars_per_1000_residents_2012,class_size_in_school_district_2011-2012,condos_as_perc_of_parcels_2012,crashes_per_1000_residents_2007-2011,culture_and_rec_spending_per_person_2012,education_spending_as_a_percent_2012,education_spending_per_resident_2012,expenditures_per_resident_2012,females_percent_in_community_2010,fire_dept_spending_as_a_percent_2012,firefighter_costs_per_resident_2012,fixed_costs_percent_2012,gun_licenses_per_1000_residents_2012,historic_places_per_10000_2013,home_schooled_per_1000_students_2011-2012,homes_built_in_39_or_before,household_member_who_is_2_races_or_more_per_1000_households_2010,households_average_size_2010,households_one-person_2010,hybrid_cars_per_1000_vehicles_2013,in_home_since_1969_or_earlier,income_average_per_resident_2010,income_change_per_resident_2007-2010,inmates_in_state_prison_per_1000_residents,liquor_licenses_per_10000_2011,median_age_2011,miles_driven_daily_per_household_05-07,minority_students_per_district_2012-2013,motorcycles_change_in_ownership_2000-2012,motorcycles_per_1000_2012,multi-generation_households_2010,police_costs_per_resident_2013,police_employees_per_10000_residents_2011,police_spending_as_a_percent_2012,population_change_1950-2010,population_change_2010-2011,presidential_fundraising_obama_vs_romney,property_crimes_per_10000_residents_2012,property_tax_change_09-13,pupils_per_cost_average_by_district_2011-2012,residential_taxes_as_percent_of_all_property_taxes_2013,saltwater_fishing_licenses_per_1000_2013,school_district_growth_09-13,single-person_households_percent_65_and_older,snowmobiles_per_10000_residents_2012,state_aid_as_a_percent_of_town_budget_2012,students_in_public_schools_2011,tax-exempt_property_2012,taxable_property_by_percent_2012,teacher_salaries_by_average_2011,teachers_percent_under_40_years_old_2011-2012,trucks_per_1000_residents_2012,violent_crimes_per_10000_residents_2012,voters_as_a_percent_of_population_2012,voters_change_in_registrations_between_1982-2012,voters_democrats_as_a_percent_2012,2020_votes,2020_biden_margin,population,vax_level"
DATASET_HEADERS_LIST = [h.strip() for h in DATASET_HEADERS.split(",")]


def read_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] == 1:
            raise ValueError
    except Exception:
        df = pd.read_csv(csv_path, sep=";", decimal=",")
    df = df.apply(pd.to_numeric, errors="ignore")
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records"), list(df.columns)


def train_test_split(examples, test_perc, seed=RANDOM_SEED):
    test_size = round(test_perc * len(examples))
    shuffled = examples.copy()
    rnd = random.Random(seed)
    rnd.shuffle(shuffled)
    return shuffled[test_size:], shuffled[:test_size]


class TreeNodeInterface:
    def classify(self, example):
        raise NotImplementedError


class DecisionNode(TreeNodeInterface):
    def __init__(self, test_attr_name, test_attr_threshold, child_lt, child_ge, child_miss):
        self.test_attr_name = test_attr_name
        self.test_attr_threshold = test_attr_threshold
        self.child_ge = child_ge
        self.child_lt = child_lt
        self.child_miss = child_miss

    def classify(self, example):
        test_val = example[self.test_attr_name]
        if test_val is None:
            return self.child_miss.classify(example)
        elif test_val < self.test_attr_threshold:
            return self.child_lt.classify(example)
        else:
            return self.child_ge.classify(example)


class LeafNode(TreeNodeInterface):
    def __init__(self, pred_class, pred_class_count, total_count):
        self.pred_class = pred_class
        self.pred_class_count = pred_class_count
        self.total_count = total_count
        self.prob = pred_class_count / total_count

    def classify(self, example):
        return self.pred_class, self.prob


class DecisionTree:
    def __init__(self, examples, id_name, class_name, min_leaf_count=1):
        self.id_name = id_name
        self.class_name = class_name
        self.min_leaf_count = min_leaf_count
        self.root = self.learn_tree(examples)

    def learn_tree(self, examples):
        attribute_set = set()
        example = examples[0]
        for attribute in example:
            if attribute != self.id_name and attribute != self.class_name:
                attribute_set.add(attribute)
        return attributeSplit(attribute_set, examples, self.min_leaf_count, self.class_name)

    def classify(self, example):
        return self.root.classify(example)


def attributeSplit(attribute_set, examples, min_leaf_count, class_name):
    attribute_name, threshold, examples_lt, examples_ge = getBestAttributeAndSplit(
        attribute_set, examples, class_name
    )
    if len(examples_ge) <= min_leaf_count or len(examples_lt) <= min_leaf_count:
        predictiveClass, predictiveClassCount = getPredictiveClass(examples, class_name)
        return LeafNode(predictiveClass, predictiveClassCount, len(examples))
    attribute_set.remove(attribute_name)
    child_lt = attributeSplit(attribute_set, examples_lt, min_leaf_count, class_name)
    child_ge = attributeSplit(attribute_set, examples_ge, min_leaf_count, class_name)
    child_miss = child_lt if len(examples_lt) >= len(examples_ge) else child_ge
    return DecisionNode(attribute_name, threshold, child_lt, child_ge, child_miss)


def getBestAttributeAndSplit(attribute_set, examples, class_label):
    maxAttribute = {"name": "", "infogain": 0.0, "threshold": None, "ex_lt": [], "ex_ge": []}
    for attribute in attribute_set:
        infoGain, threshold_, lt, ge = getInfoGain(attribute, examples, class_label)
        if infoGain > maxAttribute["infogain"]:
            maxAttribute = {
                "name": attribute,
                "infogain": infoGain,
                "threshold": threshold_,
                "ex_lt": lt,
                "ex_ge": ge,
            }
    return (maxAttribute["name"], maxAttribute["threshold"], maxAttribute["ex_lt"], maxAttribute["ex_ge"])


def getInfoGain(attribute, examples, class_label):
    maxInfoGain = 0
    threshold = 0
    lt_split = []
    ge_split = []
    min_, max_, step = getRange(attribute, examples)
    if step <= 0:
        return (0, threshold, lt_split, ge_split)
    base_entropy = entropy(examples, class_label)
    curThreshold = min_ + step
    while curThreshold < max_:
        lt, ge = splitExamplesOnAttribute(attribute, examples, curThreshold)
        pc_1 = len(lt) / len(examples)
        pc_2 = len(ge) / len(examples)
        infogain = base_entropy - ((pc_1 * entropy(lt, class_label)) + (pc_2 * entropy(ge, class_label)))
        if infogain > maxInfoGain:
            maxInfoGain = infogain
            threshold = curThreshold
            lt_split = lt
            ge_split = ge
        curThreshold += step
    return (maxInfoGain, threshold, lt_split, ge_split)


def getRange(attribute, examples):
    min_ = 1000000.0
    max_ = -1000000.0
    for ex in examples:
        if ex[attribute] is None:
            continue
        val = float(ex[attribute])
        if val < min_:
            min_ = val
        if val > max_:
            max_ = val
    step = float((max_ - min_) / 15)
    return min_, max_, step


def splitExamplesOnAttribute(attribute, examples, threshold):
    lt, ge = [], []
    for example in examples:
        if example[attribute] is None:
            continue
        elif example[attribute] >= threshold:
            ge.append(example)
        else:
            lt.append(example)
    return lt, ge


def entropy(examples, class_label):
    counts = {}
    for example in examples:
        label = example[class_label]
        counts[label] = counts.get(label, 0) + 1
    total = len(examples)
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p != 0:
            ent -= p * math.log(p, 2)
    return ent


def getPredictiveClass(examples, class_label):
    classDict = {}
    max_ = ("", 0)
    for example in examples:
        class_name = example[class_label]
        if class_name not in classDict:
            classDict[class_name] = 0
        else:
            classDict[class_name] += 1
        if classDict[class_name] > max_[1]:
            max_ = (class_name, classDict[class_name])
    return max_


def test_model(model, test_examples, label_ordering):
    correct = 0
    almost = 0
    test_act_pred = {}
    for example in test_examples:
        actual = example[model.class_name]
        pred, prob = model.classify(example)
        if pred == actual:
            correct += 1
        if abs(label_ordering.index(pred) - label_ordering.index(actual)) < 2:
            almost += 1
        test_act_pred[(actual, pred)] = test_act_pred.get((actual, pred), 0) + 1
    acc = correct / len(test_examples)
    near_acc = almost / len(test_examples)
    return acc, near_acc, test_act_pred


if __name__ == "__main__":
    path_to_csv = "town_vax_data.csv"
    examples, df_cols = read_data(path_to_csv)
    id_attr_name = "town" if "town" in df_cols else DATASET_HEADERS_LIST[0]
    class_attr_name = "vax_level" if "vax_level" in df_cols else DATASET_HEADERS_LIST[-1]
    label_ordering = ["low", "medium", "high", "very high"]
    min_examples = 10
    train_examples, test_examples = train_test_split(examples, 0.25)
    tree = DecisionTree(train_examples, id_attr_name, class_attr_name, min_examples)
    acc, _, _ = test_model(tree, test_examples, label_ordering)
    accuracy = acc
    print(f"ACCURACY={accuracy:.6f}")

# Optimization Summary
# Reduced redundant entropy computation by caching parent entropy within getInfoGain.
# Replaced list-of-examples storage in entropy with lightweight count aggregation to cut memory use.
# Avoided random.sample overhead by in-place shuffling a copied list for train/test splitting.
# Implemented robust CSV parsing with a fallback separator/decimal to prevent rework on malformed inputs.
# Converted NaN to None in a single dataframe operation to avoid per-cell checks and extra loops.