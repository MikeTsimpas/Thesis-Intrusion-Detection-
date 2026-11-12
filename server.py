from flwr.server import ServerAppComponents, ServerConfig
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from client import Net, get_parameters
from collections import OrderedDict
import torch
import flwr


params = get_parameters(Net())

class FedAvgWithGlobalAccuracy(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_accuracy = 0.0
        self.current_parameters = None   # track latest aggregated params

    def aggregate_fit(self, rnd, results, failures):
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is None:
            return None

        parameters, metrics = aggregated
        self.current_parameters = parameters   # store for aggregate_evaluate to use

        ndarrays = parameters_to_ndarrays(parameters)
        net = Net()
        state_dict = OrderedDict(
            (k, torch.tensor(v)) for k, v in zip(net.state_dict().keys(), ndarrays)
        )
        net.load_state_dict(state_dict, strict=True)

        torch.save(net.state_dict(), "global_model.pth")
        print(f"[Server] Saved global_model.pth (round {rnd})")
        return parameters, metrics
    
    def aggregate_evaluate(self, rnd, results, failures):
        if not results:
            print(
                f"\n🌍 [Server] Round {rnd} GLOBAL RESULTS\n"
                f"Global Loss: N/A (no evaluation results, failures={len(failures)})\n"
                f"Global Accuracy: N/A\n"
            )
            return float("inf"), {"global_accuracy": 0.0, "eval_failed": 1.0}

        total_correct = 0.0
        total_samples = 0
        total_loss = 0.0

        for _, res in results:
            num_samples = res.num_examples
            loss = res.loss
            accuracy = float(res.metrics.get("accuracy", 0.0))
            total_loss += loss * num_samples
            total_correct += accuracy * num_samples
            total_samples += num_samples

        global_loss = total_loss / total_samples
        global_accuracy = total_correct / total_samples

        print(
            f"\n🌍 [Server] Round {rnd} GLOBAL RESULTS\n"
            f"Global Loss: {global_loss:.6f}\n"
            f"Global Accuracy: {global_accuracy:.4f}\n"
        )

        # Save best model based on accuracy
        if global_accuracy > self.best_accuracy:
            self.best_accuracy = global_accuracy
            print(f"[Server] New best accuracy {global_accuracy:.4f} — saving best_model.pth")
            ndarrays = parameters_to_ndarrays(self.current_parameters)
            net = Net()
            state_dict = OrderedDict(
                (k, torch.tensor(v)) for k, v in zip(net.state_dict().keys(), ndarrays)
            )
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), "best_model.pth")

        return global_loss, {"global_accuracy": global_accuracy, "eval_failed": 0.0}


def server_fn() -> ServerAppComponents:
    def fit_config(rnd: int):
        return {"lr": 1e-4, "local_epochs": 1, "patience": 3, "round": rnd}
    
    strategy = FedAvgWithGlobalAccuracy(
        fraction_fit = 1.0,
        fraction_evaluate = 1.0,
        min_fit_clients = 5,
        min_evaluate_clients = 5,
        min_available_clients = 5,
        initial_parameters = ndarrays_to_parameters(params),
        on_fit_config_fn = fit_config,
    )

    print("[Server] FedAvg strategy initialized")
    return ServerAppComponents(strategy=strategy)


if __name__ == "__main__":
    print("[Server] Starting Flower server...")
    flwr.server.start_server(
        server_address="0.0.0.0:5000",
        config=ServerConfig(num_rounds=10),
        strategy=server_fn().strategy,
    )


