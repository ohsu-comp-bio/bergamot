


def dump_model(file):
    model = ml_schema.ml_schema_pb2.Model()
    with open(os.path.join(file), "rb") as f:
        model.ParseFromString(f.read())
        f.close()
    print MessageToJson(model)
