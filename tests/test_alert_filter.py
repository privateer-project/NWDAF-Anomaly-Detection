import os
import torch
import numpy as np
import unittest
from pathlib import Path

from privateer_ad.config import AlertFilterConfig
from privateer_ad.models import AlertFilterModel
from privateer_ad.train_alert_filter.feedback_collector import FeedbackCollector
from privateer_ad.train_alert_filter.alert_filter_trainer import AlertFilterTrainer

class TestAlertFilter(unittest.TestCase):
    """Test cases for the Alert Filter Model."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test data
        self.test_dir = Path('test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a small config for testing
        self.config = AlertFilterConfig(
            latent_dim=4,
            hidden_dims=[8, 4],
            dropout=0.1,
            batch_size=2,
            epochs=2
        )
        
        # Create a model
        self.model = AlertFilterModel(config=self.config)
        
        # Create some test data
        self.latent = torch.randn(5, 4)  # 5 samples, 4 features
        self.anomaly_decision = torch.tensor([1, 1, 0, 1, 0], dtype=torch.float32)
        self.reconstruction_error = torch.tensor([0.1, 0.2, 0.01, 0.15, 0.02], dtype=torch.float32)
        self.user_feedback = torch.tensor([1, 0, 0, 1, 0], dtype=torch.float32)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory and files
        for file in self.test_dir.glob('*'):
            file.unlink()
        self.test_dir.rmdir()
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        model = AlertFilterModel(config=self.config)
        self.assertIsInstance(model, AlertFilterModel)
        
        # Check that the model has the expected number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        
        # Check that the model is initialized to allow all alerts
        with torch.no_grad():
            output = model(self.latent, self.anomaly_decision, self.reconstruction_error)
            self.assertEqual(output.shape, (5, 1))
            self.assertTrue(torch.all(output > 0.5))  # All outputs should be > 0.5 (allow)
    
    def test_model_forward(self):
        """Test the forward pass of the model."""
        with torch.no_grad():
            output = self.model(self.latent, self.anomaly_decision, self.reconstruction_error)
            self.assertEqual(output.shape, (5, 1))
            self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))  # Outputs should be probabilities
    
    def test_feedback_collector(self):
        """Test the feedback collector."""
        # Create a feedback collector with a test storage path
        collector = FeedbackCollector(storage_path=self.test_dir)
        
        # Add some feedback
        for i in range(5):
            collector.add_feedback(
                latent=self.latent[i].numpy(),
                anomaly_decision=self.anomaly_decision[i].item(),
                reconstruction_error=self.reconstruction_error[i].item(),
                user_feedback=self.user_feedback[i].item()
            )
        
        # Check that the feedback was stored
        self.assertEqual(len(collector.feedback_data['user_feedback']), 5)
        
        # Check that the statistics are correct
        stats = collector.get_stats()
        self.assertEqual(stats['total'], 5)
        self.assertEqual(stats['true_positives'], 2)
        self.assertEqual(stats['false_positives'], 3)
        
        # Check that the training data can be retrieved
        training_data = collector.get_training_data()
        self.assertIsNotNone(training_data)
        self.assertEqual(len(training_data['user_feedback']), 5)
    
    def test_trainer(self):
        """Test the alert filter trainer."""
        # Create a feedback collector with test data
        collector = FeedbackCollector(storage_path=self.test_dir)
        
        # Add some feedback
        for i in range(5):
            collector.add_feedback(
                latent=self.latent[i].numpy(),
                anomaly_decision=self.anomaly_decision[i].item(),
                reconstruction_error=self.reconstruction_error[i].item(),
                user_feedback=self.user_feedback[i].item()
            )
        
        # Create a trainer
        trainer = AlertFilterTrainer(
            model=self.model,
            config=self.config
        )
        
        # Train the model
        metrics = trainer.train(
            feedback_collector=collector,
            epochs=2
        )
        
        # Check that training produced a loss
        self.assertIn('loss', metrics)
        self.assertIsInstance(metrics['loss'], float)
        
        # Save the model
        model_path = self.test_dir.joinpath('test_model.pt')
        torch.save(trainer.model.state_dict(), model_path)
        
        # Load the model
        loaded_trainer = AlertFilterTrainer.load_model(
            path=model_path,
            config=self.config
        )
        
        # Check that the loaded model works
        with torch.no_grad():
            output = loaded_trainer.model(self.latent, self.anomaly_decision, self.reconstruction_error)
            self.assertEqual(output.shape, (5, 1))
    
    def test_end_to_end(self):
        """Test the end-to-end workflow."""
        # Create a feedback collector with test data
        collector = FeedbackCollector(storage_path=self.test_dir)
        
        # Add some feedback
        for i in range(5):
            collector.add_feedback(
                latent=self.latent[i].numpy(),
                anomaly_decision=self.anomaly_decision[i].item(),
                reconstruction_error=self.reconstruction_error[i].item(),
                user_feedback=self.user_feedback[i].item()
            )
        
        # Create and train a model
        trainer = AlertFilterTrainer(config=self.config)
        trainer.train(feedback_collector=collector, epochs=10)
        
        # Make predictions with the trained model
        with torch.no_grad():
            output = trainer.model(self.latent, self.anomaly_decision, self.reconstruction_error)
            
            # Convert to binary decisions
            decisions = (output > 0.5).float()
            
            # Check that the model has learned to filter false positives
            # For samples where anomaly_decision=1 and user_feedback=0 (false positives),
            # the model should output < 0.5 (deny)
            for i in range(5):
                if self.anomaly_decision[i] == 1 and self.user_feedback[i] == 0:
                    self.assertLessEqual(output[i].item(), 0.5)
                    self.assertEqual(decisions[i].item(), 0.0)
                    
                # For samples where anomaly_decision=1 and user_feedback=1 (true positives),
                # the model should output > 0.5 (allow)
                if self.anomaly_decision[i] == 1 and self.user_feedback[i] == 1:
                    self.assertGreaterEqual(output[i].item(), 0.5)
                    self.assertEqual(decisions[i].item(), 1.0)

if __name__ == '__main__':
    unittest.main()
