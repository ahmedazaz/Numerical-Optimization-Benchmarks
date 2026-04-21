"""
optimization_modern.py

BM503 - Algoritmalarla Sayısal Yöntemler
Tek değişkenli optimizasyon (türev gerektirmeyen) yöntemleri:

- Altın-kesim araması (Golden-section search) ile minimizasyon
- Rekürsif altın-kesim araması 
- Parabolik interpolasyon ile minimizasyon

Notlar
-----
- Amaç fonksiyonu f: R -> R ve aralık [a, b] üzerinde tek modlu (unimodal) varsayımıyla çalışır.
- İterasyon geçmişi, pandas kuruluysa DataFrame olarak; değilse list[dict] olarak döndürülür.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any
import math

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

RealFunc = Callable[[float], float]


def _to_history_table(history: list[dict[str, Any]]):
    """Pandas varsa DataFrame, yoksa ham liste döndürür."""
    if pd is None:
        return history
    return pd.DataFrame(history)


@dataclass(frozen=True)
class GoldenSectionResult:
    """Altın-kesim araması sonucu."""
    x_min: float
    f_min: float
    iterations: int
    a: float
    b: float
    history: Optional[Any] = None  # DataFrame | list[dict]


def golden_section_minimize(
    f: RealFunc,
    a: float,
    b: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 200,
    return_history: bool = True,
) -> GoldenSectionResult:
    """
    Altın-kesim araması ile [a, b] aralığında minimizasyon.

    Parametreler
    ------------
    f : callable
        Minimize edilecek amaç fonksiyonu.
    a, b : float
        Arama aralığı (a < b).
    tol : float
        Durma ölçütü: aralık genişliği |b-a| <= tol.
    max_iter : int
        Maksimum iterasyon sayısı.
    return_history : bool
        True ise iterasyon geçmişi döndürülür.

    Dönüş
    -----
    GoldenSectionResult
        Sonuç nesnesinin alanları:
        - x_min : Bulunan yaklaşık minimum noktası.
        - f_min : Bu noktadaki amaç fonksiyonu değeri, f(x_min).
        - iterations : Gerçekleşen iterasyon sayısı.
        - a, b : Algoritmanın durduğu anda kalan son aralık sınırları.
        - history : return_history=True ise iterasyon geçmişi
          (pandas varsa DataFrame, yoksa list[dict]); aksi halde None.
    """
    if not (a < b):
        raise ValueError("Aralık hatası: a < b olmalıdır.")

    # invphi = (sqrt(5)-1)/2 ≈ 0.618, invphi2 = 1 - invphi ≈ 0.382
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = 1.0 - invphi

    # İlk iç noktalar
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)
    fc = f(c)
    fd = f(d)

    history: list[dict[str, Any]] = []
    k = 0

    while (b - a) > tol and k < max_iter:
        k += 1

        if return_history:
            history.append(
                {
                    "iter": k,
                    "a": a,
                    "b": b,
                    "c": c,
                    "d": d,
                    "f(c)": fc,
                    "f(d)": fd,
                    "width": b - a,
                }
            )

        # Tek modlu varsayım altında aralığı daraltma
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + invphi2 * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)

    # En iyi nokta seçimi
    if fc < fd:
        x_min, f_min = c, fc
    else:
        x_min, f_min = d, fd

    return GoldenSectionResult(
        x_min=float(x_min),
        f_min=float(f_min),
        iterations=k,
        a=float(a),
        b=float(b),
        history=_to_history_table(history) if return_history else None,
    )


@dataclass(frozen=True)
class RecursiveGoldenSectionResult:
    """
    Rekürsif altın-kesim araması sonucu.

    Alanlar
    -------
    x_min : float
        Bulunan yaklaşık minimum noktası.
    f_min : float
        Bu noktadaki amaç fonksiyonu değeri, f(x_min).
    remaining_iter : int
        Rekürsiyon durduğunda kalan derinlik hakkı.
        Başlangıçta verilen max_depth değerinden kaç adımın kullanılmadan
        kaldığını gösterir.
    """
    x_min: float
    f_min: float
    remaining_iter: int


def golden_section_minimize_recursive(
    f: RealFunc,
    a: float,
    b: float,
    *,
    tol_x: float = 1e-6,
    tol_f: float = 1e-9,
    max_depth: int = 60,
) -> RecursiveGoldenSectionResult:
    """
    Rekürsif altın-kesim araması ile minimizasyon.

    Durma ölçütü: |b-a| < tol_x ve |f(x1)-f(x2)| < tol_f veya max_depth biter.
    """
    if not (a < b):
        raise ValueError("Aralık hatası: a < b olmalıdır.")

    invphi = (math.sqrt(5.0) - 1.0) / 2.0  # ≈0.618
    x1 = a + (1.0 - invphi) * (b - a)      # ≈0.382*(b-a) + a
    x2 = a + invphi * (b - a)              # ≈0.618*(b-a) + a

    f1 = f(x1)
    f2 = f(x2)

    if max_depth <= 0 or ((b - a) < tol_x and abs(f1 - f2) < tol_f):
        if f1 <= f2:
            return RecursiveGoldenSectionResult(float(x1), float(f1), max_depth)
        return RecursiveGoldenSectionResult(float(x2), float(f2), max_depth)

    # Unimodal minimizasyonda:
    # f(x1) < f(x2) ise minimum [a, x2] aralığındadır.
    if f1 < f2:
        return golden_section_minimize_recursive(
            f, a, x2, tol_x=tol_x, tol_f=tol_f, max_depth=max_depth - 1
        )
    # f(x1) >= f(x2) ise minimum [x1, b] aralığındadır.
    return golden_section_minimize_recursive(
        f, x1, b, tol_x=tol_x, tol_f=tol_f, max_depth=max_depth - 1
    )


@dataclass(frozen=True)
class ParabolicInterpolationResult:
    """
    Parabolik interpolasyon sonucu.

    Alanlar
    -------
    x_min : float
        Bulunan yaklaşık minimum noktası.
    f_min : float
        Bu noktadaki amaç fonksiyonu değeri, f(x_min).
    iterations : int
        Gerçekleşen iterasyon sayısı.
    x1, x2, x3 : float
        Algoritmanın durduğu anda elde kalan son üç braket noktası.
    history : pandas.DataFrame | list[dict] | None
        return_history=True ise iterasyon geçmişi; aksi halde None.
    """
    x_min: float
    f_min: float
    iterations: int
    x1: float
    x2: float
    x3: float
    history: Optional[Any] = None  # DataFrame | list[dict]


def parabolic_interpolation_minimize(
    f: RealFunc,
    x1: float,
    x2: float,
    x3: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 50,
    return_history: bool = True,
) -> ParabolicInterpolationResult:
    """
    Parabolik interpolasyon ile minimizasyon (3 nokta üzerinden).

    Önkoşul (zorunlu):
    x1 < x2 < x3 ve f(x2) <= f(x1), f(x2) <= f(x3)  (minimum braketlenmiş olsun).

    Parametreler
    ------------
    f : callable
    x1, x2, x3 : float
        Başlangıç braket noktaları.
    tol : float
        Durma ölçütü: |x_new - x2| <= tol.
    max_iter : int
    return_history : bool

    Dönüş
    -----
    ParabolicInterpolationResult
    """
    if not (x1 < x2 < x3):
        raise ValueError("Nokta sıralaması hatası: x1 < x2 < x3 olmalıdır.")

    f1, f2, f3 = f(x1), f(x2), f(x3)
    if not (f2 <= f1 and f2 <= f3):
        raise ValueError(
            "Braket hatası: parabolik interpolasyon için başlangıç noktaları "
            "minimumu çevrelemelidir (f(x2) <= f(x1) ve f(x2) <= f(x3))."
        )

    history: list[dict[str, Any]] = []
    k = 0

    for k in range(1, max_iter + 1):
        # Parabol tepe noktası (vertex) hesabı
        # Formül: x = x2 - 0.5 * ((x2-x1)^2*(f2-f3) - (x2-x3)^2*(f2-f1)) / ((x2-x1)*(f2-f3) - (x2-x3)*(f2-f1))
        num = (x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)
        den = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)

        if abs(den) < 1e-15:
            # Parabolik güncelleme kararsız -> mevcut x2 ile dur
            break

        x4 = x2 - 0.5 * (num / den)
        if not (x1 < x4 < x3):
            # Parabolün tepe noktası mevcut braketin dışına çıktıysa
            # güvenilir braket korunamayacağı için mevcut x2 ile dur.
            break
        f4 = f(x4)

        if return_history:
            history.append(
                {
                    "iter": k,
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "f1": f1,
                    "f2": f2,
                    "f3": f3,
                    "x4": x4,
                    "f4": f4,
                    "step": abs(x4 - x2),
                }
            )

        # Durma ölçütü
        if abs(x4 - x2) <= tol:
            x2, f2 = x4, f4
            break

        # Braketi güncelle (minimumu çevreleyen üçlü korunacak şekilde)
        if x4 < x2:
            if f4 < f2:
                x3, f3 = x2, f2
                x2, f2 = x4, f4
            else:
                x1, f1 = x4, f4
        else:  # x4 > x2
            if f4 < f2:
                x1, f1 = x2, f2
                x2, f2 = x4, f4
            else:
                x3, f3 = x4, f4

    return ParabolicInterpolationResult(
        x_min=float(x2),
        f_min=float(f2),
        iterations=int(k),
        x1=float(x1),
        x2=float(x2),
        x3=float(x3),
        history=_to_history_table(history) if return_history else None,
    )


__all__ = [
    "GoldenSectionResult",
    "golden_section_minimize",
    "RecursiveGoldenSectionResult",
    "golden_section_minimize_recursive",
    "ParabolicInterpolationResult",
    "parabolic_interpolation_minimize",
]
