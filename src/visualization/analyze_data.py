from src.data_handling.transform import DataCleaner
from src.visualization import FeatureAnalyzer


if __name__ == '__main__':
    from src.config import ProjectPaths, MetaData
    import pandas as pd

    # Initialize config and paths
    metadata = MetaData()
    paths = ProjectPaths()

    # Create analyzer
    analyzer = FeatureAnalyzer(metadata, paths)

    # Load and analyze data
    # df = pd.read_csv(paths.processed.joinpath('test.csv'))
    features = MetaData().features
    df = pd.read_csv(paths.raw_dataset)

    df = DataCleaner(features).preprocess_data(df)
    analyzer.analyze_features(df, name='raw')