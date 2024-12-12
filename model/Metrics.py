import torch


def index_agreement_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
        index of agreement

        Willmott (1981, 1982)
        input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    ia = 1 - (torch.sum((o - s) ** 2, dim=0)) / (
        torch.sum(
            (torch.abs(s - torch.mean(o, dim=0)) +
             torch.abs(o - torch.mean(o, dim=0)))
            ** 2,
            dim=0,
        )
        + 1e-8  # avoid division by 0
    )

    return ia.mean()


def rmse_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
        rmse

        input:
        s: simulated
        o: observed
    output:
        rmse: rmse
    """
    rmse = torch.sqrt(torch.mean(torch.sum((o - s) ** 2, dim=0)))

    return rmse


def mae_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
        mae

        input:
        s: simulated
        o: observed
    output:
        rmse: rmse
    """
    mae = torch.mean(torch.abs(o - s))

    return mae