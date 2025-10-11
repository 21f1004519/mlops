from feast import Entity, FeatureView, Field
from feast.types import Float32, String
from feast.file_source import FileSource

# Define file source (use your provided file)
iris_source = FileSource(
    path="data/raw/iris_data_adapted_for_feast.csv",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define entity
iris_entity = Entity(name="iris_id", join_keys=["iris_id"])

# Define features
iris_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=None,
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="species", dtype=String),
    ],
    source=iris_source,
)
