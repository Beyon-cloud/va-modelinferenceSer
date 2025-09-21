

# Configuration Manager  ( Class 6 ConfigurationManager )
class ConfigurationManager:
    """Manage different processing configurations"""
    
    @staticmethod
    def create_high_accuracy_config(output_dir: str = "./extraction_results") -> ProcessingConfig:
        """Configuration optimized for accuracy"""
        return ProcessingConfig(
            max_retries=5,
            timeout_seconds=60,
            enable_validation=True,
            enable_post_processing=True,
            save_intermediate_results=True,
            output_directory=output_dir
        )
    
    @staticmethod
    def create_fast_processing_config() -> ProcessingConfig:
        """Configuration optimized for speed"""
        return ProcessingConfig(
            max_retries=1,
            timeout_seconds=15,
            enable_validation=False,
            enable_post_processing=False,
            save_intermediate_results=False
        )
    
    @staticmethod
    def create_development_config(output_dir: str = "./dev_extraction_results") -> ProcessingConfig:
        """Configuration for development and debugging"""
        return ProcessingConfig(
            max_retries=3,
            timeout_seconds=30,
            enable_validation=True,
            enable_post_processing=True,
            save_intermediate_results=True,
            output_directory=output_dir
        )