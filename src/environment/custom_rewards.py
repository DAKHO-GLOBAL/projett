# Créer un nouveau fichier src/environment/custom_rewards.py

import numpy as np

class GoldTradingReward:
    """Classe spécialisée pour les récompenses de trading sur l'or."""
    
    def __init__(self):
        self.previous_balance = None
        self.previous_position = 0
    
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
        Calcule une récompense adaptée au trading de l'or.
        
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
        
        # Composante principale: changement de balance
        balance_change = balance - self.previous_balance
        pnl_reward = balance_change / (self.previous_balance * 0.01) if self.previous_balance > 0 else 0
        
        # Composante d'encouragement d'action
        action_reward = 0.0
        
        # Récompenser fortement la prise de position
        if action != 0:
            action_reward += 0.5
            
            # Récompenser davantage si on prend une position quand il n'y en avait pas
            if position == 0:
                action_reward += 0.5
                
        # Pénaliser l'inaction surtout quand le marché bouge
        if action == 0 and position == 0:
            # Vérifier si le marché a un momentum
            if len(prices) >= 10:
                price_change = prices[-1] - prices[-10]
                if abs(price_change) > 0.001 * prices[-1]:  # Si le prix a changé de plus de 0.1%
                    action_reward -= 0.5  # Forte pénalité pour l'inaction en marché mobile
        
        # Combiner les récompenses
        total_reward = 0.5 * pnl_reward + 0.5 * action_reward
        
        # Mettre à jour pour le prochain calcul
        self.previous_balance = balance
        self.previous_position = position
        
        return total_reward