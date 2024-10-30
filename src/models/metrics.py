import torch


class PoseMetrics:
    @staticmethod
    def mpjpe(preds, labels):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        assert preds.shape == labels.shape
        return torch.mean(torch.norm(preds - labels, dim=len(labels.shape) - 1))

    @staticmethod
    def pa_mpjpe(preds, labels):
        """
        Procrustes Aligned MPJPE (PA-MPJPE): Applies rigid alignment (scale, rotation, translation)
        between predicted and target poses, then computes MPJPE.
        preds:  (B, N, 3)
        labels: (B, N, 3)
        """
        assert preds.shape == labels.shape
        preds_aligned = PoseMetrics.compute_similarity_transform(preds, labels)
        return PoseMetrics.mpjpe(preds_aligned, labels)

    @staticmethod
    def weighted_mpjpe(preds, labels, w):
        """
        Weighted mean per-joint position error (i.e. mean Euclidean distance)
        """
        assert preds.shape == labels.shape
        assert w.shape[0] == preds.shape[0]
        return torch.mean(w * torch.norm(preds - labels, dim=len(labels.shape) - 1))

    @staticmethod
    def mka(preds):
        """
        Mean-keypoint-acceleration (MKA) metric to measure tracking jitter
        source: https://research.facebook.com/publications/megatrack-monochrome-egocentric-articulated-hand-tracking-for-virtual-reality/ 

        params:
            preds: (batch_size, seq_len, 21, 3) predicted 3D keypoints
        return:
            mka: (batch_size) mean keypoint acceleration
        """
        # Calculate accelerations using discrete second derivative
        acc = preds[:, :-2] + preds[:, 2:] - 2 * preds[:, 1:-1]
        # Compute the norm of acceleration and take mean
        return torch.norm(acc, dim=-1).mean(dim=-1).mean(dim=-1)

    @staticmethod
    def n_mpjpe(predicted, target):
        """
        Normalized MPJPE (scale only)
        """
        assert predicted.shape == target.shape

        norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
        norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
        scale = norm_target / norm_predicted
        return PoseMetrics.mpjpe(scale * predicted, target)

    @staticmethod
    def pck(preds, labels, threshold, reference_len=None):
        """
        Percentage of Correct Keypoints (PCK)
        Args:
        - preds: (batch_size, num_joints, 3) Predicted 3D keypoints
        - labels: (batch_size, num_joints, 3) Ground truth 3D keypoints
        - threshold: Distance threshold for considering a keypoint as correctly estimated
        - reference_len: (batch_size,) or None. Optional reference distance to normalize the threshold. If None, a fixed threshold is used.
        Returns:
        - pck: (float) Percentage of keypoints that are within the threshold
        """
        # Compute Euclidean distances between corresponding joints
        assert preds.shape == labels.shape
        distances = torch.norm(preds - labels, dim=2)  # [b, n]

        # Normalize the threshold by the reference length (if provided)
        if reference_len is not None:
            threshold = threshold * reference_len.unsqueeze(1)  # Scale threshold by reference length

        # Calculate the percentage of keypoints within the threshold
        correct_keypoints = (distances <= threshold).float()  # 1 if distance <= threshold, else 0
        pck = correct_keypoints.mean()  # Mean over all keypoints and batch

        return pck
    
    @staticmethod
    def pck_auc(preds, labels, min_threshold=0, max_threshold=0.02, steps=20, reference_len=None):
        """
        Calculate both the raw AUC and the normalized AUC (norm_auc) for PCK over a range of thresholds.
        Args:
        - preds: (batch_size, num_joints, 3) Predicted 3D keypoints
        - labels: (batch_size, num_joints, 3) Ground truth 3D keypoints
        - min_threshold: Minimum threshold (e.g., 0mm or 0m)
        - max_threshold: Maximum threshold (e.g., 20mm or 0.02m)
        - steps: Number of threshold values to evaluate PCK at
        - reference_len: (batch_size,) or None. Optional reference length for normalization
        Returns:
        - auc: (float) Raw Area under the PCK curve
        - norm_auc: (float) Normalized AUC (i.e., AUC divided by the area of a perfect PCK curve)
        - pck_values: (list of floats) PCK values at each threshold
        - thresholds: (list of floats) Thresholds used for evaluation
        """
        thresholds = torch.linspace(min_threshold, max_threshold, steps)
        pck_values = []

        # Calculate PCK for each threshold
        for threshold in thresholds:
            pck_value = PoseMetrics.pck(preds, labels, threshold, reference_len)
            pck_values.append(pck_value.item())

        # Convert to a tensor and calculate raw AUC using the trapezoidal rule
        pck_values = torch.tensor(pck_values)
        auc = torch.trapz(pck_values, thresholds)

        # Calculate the area under the curve for a perfect PCK curve (where PCK is always 1.0)
        area_under_one = torch.trapz(torch.ones_like(pck_values), thresholds)

        # Calculate normalized AUC (raw AUC divided by the perfect AUC)
        norm_auc = auc / area_under_one

        return auc.item(), norm_auc.item(), pck_values.tolist(), thresholds.tolist()
    
    # src: https://github.com/geopavlakos/hamer/blob/main/hamer/utils/pose_utils.py
    @staticmethod
    def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
        """
        Computes a similarity transform (sR, t) in a batched way that takes
        a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
        where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        Args:
            S1 (torch.Tensor): First set of points of shape (B, N, 3).
            S2 (torch.Tensor): Second set of points of shape (B, N, 3).
        Returns:
            (torch.Tensor): The first set of points after applying the similarity transformation.
        """

        batch_size = S1.shape[0]
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        # 1. Remove mean.
        mu1 = S1.mean(dim=2, keepdim=True)
        mu2 = S2.mean(dim=2, keepdim=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = (X1**2).sum(dim=(1,2))

        # 3. The outer product of X1 and X2.
        K = torch.matmul(X1, X2.permute(0, 2, 1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
        U, s, V = torch.svd(K)
        Vh = V.permute(0, 2, 1)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
        Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

        # Construct R.
        R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

        # 5. Recover scale.
        trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
        scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

        # 6. Recover translation.
        t = mu2 - scale*torch.matmul(R, mu1)

        # 7. Error:
        S1_hat = scale*torch.matmul(R, S1) + t

        return S1_hat.permute(0, 2, 1)