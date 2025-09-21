

# Result Export Utilities
class ResultExporter:
    """Export extraction results to various formats"""
    
    @staticmethod
    def to_csv(results: List[ExtractionResult], output_path: str):
        """Export results to CSV"""
        try:
            import pandas as pd
            
            # Flatten results into tabular format
            records = []
            for result in results:
                if result.success and result.extracted_data:
                    record = result.extracted_data.copy()
                    record.update({
                        "extraction_success": result.success,
                        "confidence_score": result.confidence_score,
                        "validation_score": result.validation_result.get("completeness_score", 0) if result.validation_result else 0
                    })
                    records.append(record)
            
            if records:
                df = pd.DataFrame(records)
                df.to_csv(output_path, index=False)
                logger.info(f"Results exported to CSV: {output_path}")
            else:
                logger.warning("No successful results to export to CSV")
                
        except ImportError:
            logger.error("pandas required for CSV export. Install with: pip install pandas")
    
    @staticmethod
    def to_json(results: List[ExtractionResult], output_path: str):
        """Export results to JSON"""
        export_data = []
        
        for result in results:
            result_data = {
                "success": result.success,
                "extracted_data": result.extracted_data,
                "validation_result": result.validation_result,
                "confidence_score": result.confidence_score,
                "errors": result.errors,
                "warnings": result.warnings,
                "processing_metadata": result.processing_metadata
            }
            export_data.append(result_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to JSON: {output_path}")