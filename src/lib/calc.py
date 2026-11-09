import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def fit_models(raw_data):
    # Svelte에서 전달한 JSON 데이터
    data = json.loads(raw_data)

    df = pd.DataFrame(data)

    required_columns = ['Budget', 'CVR', 'Users']

    if not all(col in df.columns for col in required_columns):
        raise ValueError("필수 데이터 항목이 누락되었습니다.")

    if len(df) < 5:
        raise ValueError("회귀 분석을 위해 최소 5개의 데이터가 필요합니다.")

    # 숫자형 변환
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if not np.isfinite(df[required_columns].to_numpy()).all():
        raise ValueError("숫자가 아닌 값이 포함되어 있습니다.")

    if (df['Budget'] <= 0).any():
        raise ValueError("Budget은 0보다 커야 합니다.")

    if ((df['CVR'] < 0) | (df['CVR'] > 100)).any():
        raise ValueError("CVR은 0~100 사이여야 합니다.")

    if (df['Users'] < 0).any():
        raise ValueError("Users는 0 이상이어야 합니다.")

    # CVR은 화면에서 퍼센트로 입력받으므로 소수로 변환
    df['CVR'] = df['CVR'] / 100

    X = df[['Budget', 'CVR']]
    y = df['Users']

    # -----------------------------
    # 1. 선형 회귀
    # -----------------------------

    linear_model = LinearRegression()
    linear_model.fit(X, y)

    linear_result = {
        'w_budget': float(linear_model.coef_[0]),
        'w_cvr': float(linear_model.coef_[1]),
        'intercept': float(linear_model.intercept_),
        'r2': float(linear_model.score(X, y))
    }

    # -----------------------------
    # 2. 2차 다항 회귀
    # -----------------------------

    poly = PolynomialFeatures(
        degree=2,
        include_bias=False
    )

    X_budget_poly = poly.fit_transform(
        df[['Budget']]
    )

    X_nonlinear = np.hstack((
        X_budget_poly,
        df[['CVR']].to_numpy()
    ))

    poly_model = LinearRegression()
    poly_model.fit(X_nonlinear, y)

    poly_result = {
        'w1': float(poly_model.coef_[0]),
        'w2': float(poly_model.coef_[1]),
        'w3': float(poly_model.coef_[2]),
        'intercept': float(poly_model.intercept_),
        'r2': float(poly_model.score(X_nonlinear, y))
    }

    return {
        'linear': linear_result,
        'poly': poly_result,
        'sample_count': len(df)
    }