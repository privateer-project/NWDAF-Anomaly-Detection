from data_handling.transform import DataCleaner
from visualization import FeatureAnalyzer

if __name__ == '__main__':
    from config import Paths, MetaData
    import pandas as pd

    # Initialize config and paths
    metadata = MetaData()
    paths = Paths()

    # Create analyzer
    analyzer = FeatureAnalyzer(metadata, paths)

    # Load and analyze data
    # df = pd.read_csv(paths.processed.joinpath('test.csv'))
    features = MetaData().features
    df = pd.read_csv(paths.raw_dataset)

    df = DataCleaner(features).preprocess_data(df)
    analyzer.analyze_features(df, name='raw')