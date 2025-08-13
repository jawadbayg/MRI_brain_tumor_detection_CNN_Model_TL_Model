def train_model(model, train_gen, val_gen, epochs=15):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )
    return history
