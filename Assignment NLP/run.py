#Phase 1 dataset loader and validation
# from src.dataset import DatasetLoader
# from src.visualization import DatasetVisualizer
# from src.utils import print_heading


# def main():

#     loader = DatasetLoader()

#     visualizer = DatasetVisualizer()

#     train, validation, test = loader.load_data()

#     print_heading("TRAIN DATASET")

#     loader.dataset_info(train, "Train")

#     print("\nMissing Values")

#     print(loader.missing_values(train))

#     print("\nDuplicate Rows")

#     print(loader.duplicate_rows(train))

#     print("\nClass Distribution")

#     print(loader.class_distribution(train))

#     print_heading("VALIDATION DATASET")

#     loader.dataset_info(validation, "Validation")

#     print_heading("TEST DATASET")

#     loader.dataset_info(test, "Test")

#     full_dataset = loader.combine_datasets()

#     print_heading("FULL DATASET")

#     print(full_dataset.shape)

#     visualizer.plot_class_distribution(full_dataset)

#     visualizer.plot_text_length(full_dataset)

#     print_heading("VALIDATION DATASET")

#     loader.dataset_info(validation, "Validation")

#     print("\nMissing Values")
#     print(loader.missing_values(validation))

#     print("\nDuplicate Rows")
#     print(loader.duplicate_rows(validation))

#     print("\nClass Distribution")
#     print(loader.class_distribution(validation))


#     print_heading("TEST DATASET")

#     loader.dataset_info(test, "Test")

#     print("\nMissing Values")
#     print(loader.missing_values(test))

#     print("\nDuplicate Rows")
#     print(loader.duplicate_rows(test))

#     print("\nClass Distribution")
#     print(loader.class_distribution(test))

#     print_heading("FULL DATASET")

#     print("Shape:", full_dataset.shape)

#     print("\nUnique Classes:")
#     print(full_dataset["label"].unique())

#     print("\nNumber of Classes:")
#     print(full_dataset["label"].nunique())

#     print("\nOverall Class Distribution")
#     print(loader.class_distribution(full_dataset))


# if __name__ == "__main__":
#     main()

#phase2 dataset Exploratory Data Analysis and Data Cleaning
# from src.pipeline import Pipeline


# def main():

#     pipeline = Pipeline()

#     pipeline.run()


# if __name__ == "__main__":

#     main()

#phase 3 Stemming & Lemmatization.
from src.pipeline import Pipeline


def main():

    pipeline = Pipeline()

    pipeline.run()


if __name__ == "__main__":
    main()