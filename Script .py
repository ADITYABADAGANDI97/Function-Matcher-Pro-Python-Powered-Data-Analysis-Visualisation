import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from bokeh.plotting import figure, show
from bokeh.layouts import column
from bokeh.palettes import Spectral4
import unittest
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Exceptions
class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass

class DatabaseError(Exception):
    """Raised when database operations fail"""
    pass

# Base class for SQLAlchemy models
Base = declarative_base()

class DataModel(Base):
    """Base model for all data tables"""
    __abstract__ = True
    
    x = Column(Float, primary_key=True)
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(x={self.x})>"

class TrainingData(DataModel):
    """Model for training data table from train.csv"""
    __tablename__ = 'training_data'
    
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)

class IdealFunctions(DataModel):
    """Model for ideal functions table from ideal.csv"""
    __tablename__ = 'ideal_functions'
    
    # Create columns y1 through y50
    columns = {f'y{i}': Column(Float) for i in range(1, 51)}
    locals().update(columns)

class TestResults(DataModel):
    """Model for test results table from test.csv with additional analysis"""
    __tablename__ = 'test_results'
    
    y = Column(Float)
    delta_y = Column(Float)
    ideal_func_num = Column(Integer)

class DataAnalyzer:
    """Main class for data analysis operations"""
    
    def __init__(self, db_path: str = 'analysis.db'):
        """
        Initialize the DataAnalyzer with database connection
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        
    def load_training_data(self, train_file: str) -> None:
        """
        Load training data from train.csv into database
        
        Args:
            train_file (str): Path to train.csv file
        
        Raises:
            DataValidationError: If data validation fails
        """
        try:
            # Read training data
            data = pd.read_csv(train_file)
            expected_columns = ['x', 'y1', 'y2', 'y3', 'y4']
            
            if not all(col in data.columns for col in expected_columns):
                raise DataValidationError(f"Missing columns in {train_file}. Expected: {expected_columns}")
            
            # Convert DataFrame to database records
            for _, row in data.iterrows():
                training_data = TrainingData(
                    x=row['x'],
                    y1=row['y1'],
                    y2=row['y2'],
                    y3=row['y3'],
                    y4=row['y4']
                )
                self.session.add(training_data)
            
            self.session.commit()
            logger.info(f"Successfully loaded training data from {train_file}")
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to load training data: {str(e)}")
            
    def load_ideal_functions(self, ideal_file: str) -> None:
        """
        Load ideal functions from ideal.csv into database
        
        Args:
            ideal_file (str): Path to ideal.csv file
        
        Raises:
            DataValidationError: If data validation fails
        """
        try:
            # Read ideal functions data
            data = pd.read_csv(ideal_file)
            expected_columns = ['x'] + [f'y{i}' for i in range(1, 51)]
            
            if not all(col in data.columns for col in expected_columns):
                raise DataValidationError(f"Missing columns in {ideal_file}")
            
            # Convert DataFrame to database records
            for _, row in data.iterrows():
                ideal_func = IdealFunctions(
                    x=row['x'],
                    **{f'y{i}': row[f'y{i}'] for i in range(1, 51)}
                )
                self.session.add(ideal_func)
                
            self.session.commit()
            logger.info(f"Successfully loaded ideal functions from {ideal_file}")
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to load ideal functions: {str(e)}")
            
    def process_test_data(self, test_file: str, condition_criterion: float) -> None:
        """
        Process test data from test.csv and match to ideal functions
        
        Args:
            test_file (str): Path to test.csv file
            condition_criterion (float): Maximum allowed deviation
            
        Raises:
            DataValidationError: If data validation fails
        """
        try:
            # Read test data
            test_data = pd.read_csv(test_file)
            if not all(col in test_data.columns for col in ['x', 'y']):
                raise DataValidationError("Test data must contain 'x' and 'y' columns")
            
            # Get ideal functions data
            ideal_funcs = pd.read_sql('select * from ideal_functions', self.engine)
            
            # Process each test point
            for _, row in test_data.iterrows():
                x, y = row['x'], row['y']
                
                # Find best matching ideal function
                min_deviation = float('inf')
                best_func_num = None
                
                # Check all 50 ideal functions
                for i in range(1, 51):
                    # Find closest x-value in ideal function
                    ideal_y = ideal_funcs[f'y{i}'].iloc[
                        (ideal_funcs['x'] - x).abs().idxmin()
                    ]
                    deviation = abs(ideal_y - y)
                    
                    if deviation < min_deviation and deviation <= condition_criterion:
                        min_deviation = deviation
                        best_func_num = i
                
                if best_func_num is not None:
                    result = TestResults(
                        x=x,
                        y=y,
                        delta_y=min_deviation,
                        ideal_func_num=best_func_num
                    )
                    self.session.add(result)
            
            self.session.commit()
            logger.info("Successfully processed test data")
            
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(f"Failed to process test data: {str(e)}")
            
    def visualize_results(self) -> None:
        """
        Create visualization of results using Bokeh
        """
        # Fetch all data
        training_data = pd.read_sql('select * from training_data', self.engine)
        ideal_funcs = pd.read_sql('select * from ideal_functions', self.engine)
        test_results = pd.read_sql('select * from test_results', self.engine)
        
        # Create main plot
        p = figure(title="Data Analysis Results", 
                  x_axis_label='X', 
                  y_axis_label='Y',
                  width=800, 
                  height=600)
        
        # Plot training data
        colors = ['blue', 'green', 'red', 'purple']
        for i, color in zip(range(1, 5), colors):
            p.line(training_data['x'], training_data[f'y{i}'], 
                  legend_label=f'Training Function {i}', 
                  color=color, 
                  line_dash='dashed',
                  line_width=2)
        
        # Plot matched ideal functions
        unique_funcs = test_results['ideal_func_num'].unique()
        for func_num in unique_funcs:
            p.line(ideal_funcs['x'], ideal_funcs[f'y{func_num}'],
                  legend_label=f'Ideal Function {func_num}',
                  color='black',
                  line_width=1)
        
        # Plot test points
        p.circle(test_results['x'], test_results['y'],
                legend_label='Test Points',
                color='red',
                size=8)
        
        # Configure legend
        p.legend.click_policy = "hide"
        p.legend.location = "top_right"
        
        # Show plot
        show(p)
        logger.info("Visualization completed")

class TestDataAnalyzer(unittest.TestCase):
    """Unit tests for DataAnalyzer"""
    
    def setUp(self):
        """Set up test environment"""
        self.analyzer = DataAnalyzer('test.db')
        
    def test_load_training_data(self):
        """Test loading training data"""
        # Create sample training data
        sample_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1, 2, 3],
            'y2': [2, 3, 4],
            'y3': [3, 4, 5],
            'y4': [4, 5, 6]
        })
        sample_data.to_csv('sample_train.csv', index=False)
        
        self.analyzer.load_training_data('sample_train.csv')
        loaded_data = pd.read_sql('select * from training_data', self.analyzer.engine)
        self.assertEqual(len(loaded_data), 3)
        
    def test_invalid_training_data(self):
        """Test loading invalid training data"""
        # Create invalid training data
        invalid_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1, 2, 3]  # Missing columns
        })
        invalid_data.to_csv('invalid_train.csv', index=False)
        
        with self.assertRaises(DataValidationError):
            self.analyzer.load_training_data('invalid_train.csv')
            
    def tearDown(self):
        """Clean up test environment"""
        import os
        if os.path.exists('test.db'):
            os.remove('test.db')
        if os.path.exists('sample_train.csv'):
            os.remove('sample_train.csv')
        if os.path.exists('invalid_train.csv'):
            os.remove('invalid_train.csv')

if __name__ == '__main__':
    # Example usage
    analyzer = DataAnalyzer()
    
    try:
        # Load training data
        analyzer.load_training_data('train.csv')
        
        # Load ideal functions
        analyzer.load_ideal_functions('ideal.csv')
        
        # Process test data with condition criterion
        analyzer.process_test_data('test.csv', condition_criterion=0.1)
        
        # Visualize results
        analyzer.visualize_results()
        
    except (DataValidationError, DatabaseError) as e:
        logger.error(f"Error during analysis: {str(e)}")
        
    finally:
        analyzer.session.close()