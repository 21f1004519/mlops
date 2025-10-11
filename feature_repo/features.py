from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.bigquery_source import BigQuerySource
from feast.types import Float32, String

iris_source = BigQuerySource(
    table="heroic-throne-473405-m8.iris_week3ga.iris_data",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

iris_entity = Entity(name="iris_id", join_keys=["iris_id"])

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
    online=True,
)
