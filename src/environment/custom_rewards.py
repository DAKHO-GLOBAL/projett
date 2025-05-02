# Mise à jour de la classe GoldTradingReward dans src/environment/custom_rewards.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class GoldTradingReward:
    """Fonction de récompense optimisée pour le trading de l'or."""
    
    def __init__(self):
        self.previous_balance = None
        self.previous_position = 0
        self.position_duration = 0
        self.price_history = []
        self.reward_history = []
        self.max_position_duration = 288  # Max 24 heures en timeframe M5 (12 * 24)
    

    # Correction dans src/environment/custom_rewards.py

    def calculate_reward(
        self,
        action: int,
        position: int,
        unrealized_pnl: float,
        realized_pnl: float,
        balance: float,
        equity: float,
        prices: np.ndarray
    ) -> float:
        """
        Calcule une récompense avancée pour le trading de l'or.
        
        Args:
            action: Action prise (-1: vente, 0: attente, 1: achat)
            position: Position actuelle (-1: short, 0: flat, 1: long)
            unrealized_pnl: Profit/perte non réalisé
            realized_pnl: Profit/perte réalisé
            balance: Solde du compte
            equity: Équité du compte
            prices: Tableau des prix récents
            
        Returns:
            float: Récompense calculée
        """
        # Initialiser le solde précédent si nécessaire
        if self.previous_balance is None:
            self.previous_balance = balance
        
        # Ajouter le prix actuel à l'historique
        if len(prices) > 0:
            self.price_history.append(prices[-1])
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        # Mettre à jour la durée de position
        if self.previous_position != 0 and position != 0 and position == self.previous_position:
            self.position_duration += 1
        else:
            self.position_duration = 0 if position == 0 else 1
        
        # --- Composantes de la récompense ---
        
        # 1. Récompense de P&L (40% du poids total)
        balance_change = balance - self.previous_balance
        pnl_ratio = balance_change / (self.previous_balance * 0.01) if self.previous_balance > 0 else 0
        
        if balance_change > 0:
            # Récompense positive pour les gains
            pnl_reward = pnl_ratio * 1.5  # Amplifier les résultats positifs
        else:
            # Pénalité pour les pertes, mais légèrement atténuée
            pnl_reward = pnl_ratio * 1.0
            
        # 2. Récompense d'action intelligente (25% du poids total)
        action_reward = 0.0
        
        # Récompenser la prise de position quand il y a tendance
        if action != 0 and len(prices) >= 5:
            # Tendance à court terme
            short_trend = prices[-1] - prices[-3]
            
            # Si action suit la tendance du marché
            if (action == 1 and short_trend > 0) or (action == -1 and short_trend < 0):
                action_reward += 0.3  # Action dans le sens de la tendance
            else:
                action_reward -= 0.1  # Action contre-tendance
                
        # 3. Récompense de gestion du temps (15% du poids total)
        time_reward = 0.0
        
        # Pénalité pour les positions tenues trop longtemps sans profit
        if position != 0 and self.position_duration > 72:  # Plus de 6 heures
            if unrealized_pnl <= 0:
                # Pénalité croissante pour les positions perdantes maintenues longtemps
                time_reward -= 0.1 * (self.position_duration / 72)
            else:
                # Légère pénalité pour les positions gagnantes maintenues très longtemps
                if self.position_duration > 144:  # Plus de 12 heures
                    time_reward -= 0.05 * ((self.position_duration - 144) / 72)
        
        # 4. Récompense de volatilité (20% du poids total)
        volatility_reward = 0.0
        
        if len(prices) >= 10:
            # CORRECTION: Calcul de la volatilité récente
            # Utiliser le slicing correct pour éviter le problème de dimensions
            price_changes = np.diff(prices[-10:]) / prices[-10:-1]  # Corrigé ici
            recent_volatility = np.std(price_changes) * 100  # En pourcentage
            
            # L'or est plus tradable dans les périodes de volatilité modérée
            if 0.05 < recent_volatility < 0.2:  # Plage de volatilité optimale pour l'or
                # Récompenser les actions dans les périodes de volatilité optimale
                if action != 0:
                    volatility_reward += 0.3
                elif position == 0:
                    volatility_reward -= 0.1  # Pénalité légère pour l'inaction en bonne volatilité
            elif recent_volatility > 0.3:
                # Volatilité excessive - récompenser la prudence
                if action == 0 and position == 0:
                    volatility_reward += 0.2  # Récompenser l'attente pendant forte volatilité
                elif action != 0 and position == 0:
                    volatility_reward -= 0.2  # Pénaliser la prise de risque en forte volatilité
        
        # Combinaison pondérée des récompenses
        total_reward = (
            0.40 * pnl_reward +
            0.25 * action_reward +
            0.15 * time_reward +
            0.20 * volatility_reward
        )
        
        # Normalisation et limite pour éviter les valeurs extrêmes
        total_reward = max(min(total_reward, 2.0), -2.0)
        
        # Conserver l'historique des récompenses pour l'analyse
        self.reward_history.append(total_reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
        
        # Mettre à jour pour le prochain calcul
        self.previous_balance = balance
        self.previous_position = position
        
        return total_reward

