# Classe de gestion des risques optimisée pour l'or
# À placer dans src/risk_management/gold_risk_manager.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple

from src.risk_management.stop_manager import StopManager
from src.risk_management.position_sizer import PositionSizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class GoldRiskManager:
    """Gestionnaire de risque spécialisé pour le trading de l'or."""
    
    def __init__(self, config: Dict):
        """
        Initialise le gestionnaire de risque pour l'or.
        
        Args:
            config: Dictionnaire de configuration
        """
        self.config = config
        
        # Instancier les gestionnaires de base
        self.stop_manager = StopManager(config)
        self.position_sizer = PositionSizer(config)
        
        # Paramètres spécifiques à l'or
        gold_config = config.get('gold_trading', {})
        self.volatility_adjustement = gold_config.get('volatility_adjustment', True)
        self.session_based_sizing = gold_config.get('session_based_sizing', True)
        self.max_daily_trades = gold_config.get('max_daily_trades', 5)
        self.max_concurrent_trades = gold_config.get('max_concurrent_trades', 2)
        self.minimum_trades_per_week = gold_config.get('minimum_trades_per_week', 10)
        
        # Historique des trades et performances
        self.trade_history = []
        self.current_day_trades = 0
        self.current_week_trades = 0
        self.last_day = None
        self.last_week = None
        
        logger.info("GoldRiskManager initialisé avec configuration spécifique à l'or")
    
    def calculate_position_size(
        self,
        action: int,
        balance: float,
        price: float,
        volatility: float,
        current_time: pd.Timestamp,
        atr: Optional[float] = None
    ) -> float:
        """
        Calcule la taille de position adaptée pour l'or.
        
        Args:
            action: Action (-1 pour vente, 1 pour achat)
            balance: Solde du compte
            price: Prix actuel
            volatility: Volatilité actuelle (ex: ATR normalisé)
            current_time: Horodatage actuel
            atr: ATR brut (si disponible)
            
        Returns:
            float: Taille de position en lots
        """
        # Position de base calculée par le position sizer standard
        base_size = self.position_sizer.calculate_position_size(
            action=action,
            balance=balance,
            price=price,
            symbol="XAUUSD",
            atr=atr
        )
        
        # Facteurs d'ajustement
        volatility_factor = 1.0
        session_factor = 1.0
        trade_frequency_factor = 1.0
        
        # 1. Ajustement selon la volatilité
        if self.volatility_adjustement and volatility is not None:
            # Réduire la taille pour une forte volatilité, augmenter pour une faible
            if volatility > 0.8:  # Forte volatilité (80% au-dessus de la normale)
                volatility_factor = 0.7  # Réduire de 30%
            elif volatility < 0.3:  # Faible volatilité
                volatility_factor = 0.8  # Réduire de 20% (éviter de piéger dans des ranges)
            else:
                # Volatilité optimale (entre 30% et 80% de la normale)
                volatility_factor = 1.1  # Augmenter légèrement
        
        # 2. Ajustement selon la session de trading
        if self.session_based_sizing and isinstance(current_time, pd.Timestamp):
            hour = current_time.hour
            day_of_week = current_time.dayofweek
            
            # Ajustement selon l'heure (sessions optimales pour l'or)
            if 8 <= hour <= 11:  # Session européenne du matin
                session_factor = 1.1
            elif 13 <= hour <= 16:  # Chevauchement Europe-US
                session_factor = 1.2  # Meilleure liquidité, plus forte volatilité
            elif 20 <= hour <= 22:  # Fin de session US
                session_factor = 0.8  # Plus risqué
            elif hour < 3 or hour > 22:  # Nuit, faible liquidité
                session_factor = 0.6  # Réduire significativement
            
            # Ajustement selon le jour de la semaine
            if day_of_week == 0:  # Lundi
                session_factor *= 0.9  # Souvent moins de tendance
            elif day_of_week == 2:  # Mercredi
                session_factor *= 1.1  # Souvent plus d'action (FOMC)
            elif day_of_week == 4 and hour >= 15:  # Vendredi après-midi
                session_factor *= 0.6  # Risque de gaps de week-end
        
        # 3. Ajustement selon la fréquence de trading
        # Vérifier si on a changé de jour
        current_day = current_time.date() if isinstance(current_time, pd.Timestamp) else None
        current_week = current_time.isocalendar()[1] if isinstance(current_time, pd.Timestamp) else None
        
        if current_day != self.last_day:
            self.current_day_trades = 0
            self.last_day = current_day
        
        if current_week != self.last_week:
            self.current_week_trades = 0
            self.last_week = current_week
        
        # Limiter le nombre de trades par jour
        if self.current_day_trades >= self.max_daily_trades:
            trade_frequency_factor = 0.1  # Presque pas de trading
        elif self.current_day_trades >= self.max_daily_trades - 1:
            trade_frequency_factor = 0.5  # Réduire significativement
        
        # Encourager le trading si on a fait peu de trades cette semaine
        if self.current_week_trades < self.minimum_trades_per_week // 2:
            # Milieu de semaine et peu de trades
            if 1 <= current_time.dayofweek <= 3:
                trade_frequency_factor *= 1.2  # Encourager plus de trades
        
        # Calculer la taille finale
        final_size = base_size * volatility_factor * session_factor * trade_frequency_factor
        
        # Garantir une taille minimum
        minimum_size = 0.01  # 0.01 lot minimum pour l'or
        final_size = max(final_size, minimum_size)
        
        # Arrondir à la précision de lot standard pour l'or
        lot_precision = 0.01
        final_size = round(final_size / lot_precision) * lot_precision
        
        logger.info(f"GoldRiskManager: Taille calculée={final_size} (base={base_size}, "
                   f"vol_factor={volatility_factor}, session_factor={session_factor}, "
                   f"freq_factor={trade_frequency_factor})")
        
        return final_size
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: int,
        volatility: float,
        atr: Optional[float] = None,
        recent_prices: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Calcule les niveaux de stop loss optimisés pour l'or.
        
        Args:
            entry_price: Prix d'entrée
            direction: Direction (1 pour long, -1 pour short)
            volatility: Mesure de volatilité normalisée
            atr: ATR brut (si disponible)
            recent_prices: Données de prix récentes
            
        Returns:
            Dict: Niveaux de stop loss et informations associées
        """
        # Utiliser le StopManager de base pour le calcul initial
        base_stop = self.stop_manager.calculate_stop_loss(
            entry_price=entry_price,
            direction=direction,
            atr=atr,
            recent_prices=recent_prices
        )
        
        # Calculer la distance du stop en points
        stop_distance = abs(entry_price - base_stop)
        
        # Ajuster la distance en fonction de la volatilité spécifique à l'or
        if volatility > 0.8:  # Forte volatilité
            # Élargir le stop pour éviter d'être shakeout
            adjusted_distance = stop_distance * 1.2
        elif volatility < 0.4:  # Faible volatilité
            # Réduire légèrement le stop
            adjusted_distance = stop_distance * 0.9
        else:
            # Volatilité normale
            adjusted_distance = stop_distance
        
        # Appliquer la distance ajustée
        if direction == 1:  # Long
            adjusted_stop = entry_price - adjusted_distance
        else:  # Short
            adjusted_stop = entry_price + adjusted_distance
        
        # Vérifier les niveaux psychologiques pour l'or
        # L'or respecte souvent les niveaux ronds
        round_levels = []
        
        # Niveaux ronds aux 50$
        base_level = (entry_price // 50) * 50
        for i in range(-2, 3):
            round_levels.append(base_level + i * 50)
        
        # Trouver le niveau rond le plus proche en direction du stop loss
        nearest_level = None
        nearest_distance = float('inf')
        
        for level in round_levels:
            # Vérifier si le niveau est dans la bonne direction par rapport à l'entrée
            if (direction == 1 and level < entry_price) or (direction == -1 and level > entry_price):
                distance = abs(level - adjusted_stop)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_level = level
        
        # Si un niveau rond est proche du stop calculé (±10), utiliser ce niveau
        if nearest_level is not None and nearest_distance < 10:
            final_stop = nearest_level
            stop_type = "psychological_level"
        else:
            final_stop = adjusted_stop
            stop_type = "adjusted_atr"
        
        # Calculer l'exposition au risque
        risk_amount = abs(entry_price - final_stop)
        risk_percent = (risk_amount / entry_price) * 100
        
        return {
            'stop_loss': final_stop,
            'base_stop': base_stop,
            'stop_type': stop_type,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'volatility_factor': volatility
        }
    
    def calculate_take_profit_levels(
        self,
        entry_price: float,
        stop_loss: float,
        direction: int,
        volatility: float
    ) -> Dict:
        """
        Calcule les niveaux de prise de profit optimisés pour l'or.
        
        Args:
            entry_price: Prix d'entrée
            stop_loss: Niveau de stop loss
            direction: Direction (1 pour long, -1 pour short)
            volatility: Mesure de volatilité
            
        Returns:
            Dict: Niveaux de prise de profit et informations associées
        """
        # Calculer la distance du stop
        stop_distance = abs(entry_price - stop_loss)
        
        # Risque de base (1R)
        r1 = stop_distance
        
        # Adapter le ratio risque/récompense selon la volatilité
        if volatility > 0.8:  # Forte volatilité = plus de potentiel
            r_ratio_1 = 1.5  # Premier TP à 1.5x la distance du stop
            r_ratio_2 = 2.5  # Deuxième TP à 2.5x
            r_ratio_3 = 4.0  # Troisième TP à 4x
        elif volatility < 0.4:  # Faible volatilité = moins de potentiel
            r_ratio_1 = 1.2  # Premier TP plus proche
            r_ratio_2 = 2.0
            r_ratio_3 = 3.0
        else:  # Volatilité normale
            r_ratio_1 = 1.3
            r_ratio_2 = 2.2
            r_ratio_3 = 3.5
        
        # Calculer les distances en R
        tp1_distance = r1 * r_ratio_1
        tp2_distance = r1 * r_ratio_2
        tp3_distance = r1 * r_ratio_3
        
        # Calculer les niveaux de prix
        if direction == 1:  # Long
            tp1 = entry_price + tp1_distance
            tp2 = entry_price + tp2_distance
            tp3 = entry_price + tp3_distance
        else:  # Short
            tp1 = entry_price - tp1_distance
            tp2 = entry_price - tp2_distance
            tp3 = entry_price - tp3_distance
        
        # Sorties partielles recommandées (en pourcentage de la position)
        exit_percentages = [0.4, 0.3, 0.2]  # 90% au total, 10% runner
        
        # Ajuster vers les niveaux psychologiques (multiples de 25$)
        tp1 = round(tp1 / 25) * 25
        tp2 = round(tp2 / 25) * 25
        tp3 = round(tp3 / 25) * 25
        
        return {
            'take_profit_levels': [tp1, tp2, tp3],
            'exit_percentages': exit_percentages,
            'r_values': [r_ratio_1, r_ratio_2, r_ratio_3],
            'initial_stop_distance': stop_distance,
            'risk_reward_ratio': r_ratio_3  # Ratio final
        }
    
    def should_trade_now(
        self,
        current_time: pd.Timestamp,
        market_state: Dict,
        account_state: Dict
    ) -> Tuple[bool, str]:
        """
        Détermine s'il est opportun de trader l'or maintenant.
        
        Args:
            current_time: Horodatage actuel
            market_state: État du marché (volatilité, tendance, etc.)
            account_state: État du compte (balance, drawdown, etc.)
            
        Returns:
            Tuple[bool, str]: (Doit trader, raison)
        """
        # 1. Vérifier les limites de trading quotidien
        if self.current_day_trades >= self.max_daily_trades:
            return False, "Limite quotidienne de trades atteinte"
        
        # 2. Vérifier le nombre de positions concurrentes
        if account_state.get('open_positions', 0) >= self.max_concurrent_trades:
            return False, "Nombre maximum de positions atteint"
        
        # 3. Vérifier la session de trading
        if isinstance(current_time, pd.Timestamp):
            hour = current_time.hour
            day = current_time.dayofweek
            
            # Sessions à éviter pour l'or
            if hour < 3 or hour > 22:
                return False, "Hors des heures de trading optimales pour l'or"
            
            # Vendredi soir - éviter les positions overnight
            if day == 4 and hour >= 20:
                return False, "Vendredi soir - risque de gap de weekend"
            
            # Début de semaine, faible liquidité
            if day == 0 and hour < 8:
                return False, "Début de semaine, attendre plus de clarté"
        
        # 4. Vérifier la volatilité
        volatility = market_state.get('volatility', 0.5)
        if volatility < 0.2:
            return False, "Volatilité trop faible pour trader l'or efficacement"
        
        # 5. Vérifier le drawdown du compte
        max_drawdown = account_state.get('max_drawdown', 0)
        if max_drawdown > 0.15:  # 15% de drawdown
            return False, "Drawdown trop important, pause de trading recommandée"
        
        # 6. Vérifier le ratio gain/perte récent
        win_rate = account_state.get('win_rate', 0.5)
        if win_rate < 0.3 and self.current_day_trades > 2:
            return False, "Performance récente faible, reconsidérer la stratégie"
        
        # Par défaut, autoriser le trading
        return True, "Conditions favorables pour le trading de l'or"