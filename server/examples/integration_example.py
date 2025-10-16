"""
Integration Script: Add Tampering Localization to DocuForge API
================================================================
This script shows how to integrate the tampering localization module
into your existing DocuForge forgery detection system.
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import tampering localization
from detection.tampering_localization import detect_tampering_hybrid, DocumentTamperingDetector

# Import your existing model loading functions
try:
    from models.predict import load_model, predict_image
except ImportError:
    print("Warning: Could not import predict module")


class EnhancedDocumentAnalyzer:
    """
    Enhanced document analyzer with tampering localization capabilities.
    Extends the basic forgery detection with detailed localization.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the enhanced analyzer.
        
        Args:
            model_path: Path to trained model
            device: Device for inference (auto-detected if None)
        """
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / "models" / "saved_models" / "best_model.pth")
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def analyze_document(self, image_path: str, 
                        enable_localization: bool = True,
                        sensitivity: float = 0.5,
                        save_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive document analysis with optional localization.
        
        Args:
            image_path: Path to document image
            enable_localization: Whether to perform tampering localization
            sensitivity: Threshold for tampering detection (0-1)
            save_results: Whether to save visualization results
        
        Returns:
            Dictionary with analysis results
        """
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\n{'='*70}")
        print(f"Analyzing: {os.path.basename(image_path)}")
        print(f"{'='*70}\n")
        
        # Basic forgery detection (your existing prediction)
        try:
            # If you have a predict function, use it
            basic_prediction = self._basic_prediction(image_path)
            print(f"✓ Basic forgery detection: {basic_prediction['class']}")
            print(f"  Confidence: {basic_prediction['confidence']:.2%}")
        except Exception as e:
            print(f"⚠️  Basic prediction unavailable: {e}")
            basic_prediction = {"class": "unknown", "confidence": 0.5}
        
        result = {
            "image_path": image_path,
            "basic_detection": basic_prediction,
        }
        
        # Enhanced localization (if enabled)
        if enable_localization:
            print(f"\n🔬 Performing detailed tampering localization...")
            
            try:
                localization_result = detect_tampering_hybrid(
                    image_path=image_path,
                    model=self.model,
                    device=self.device,
                    save_results=save_results,
                    sensitivity=sensitivity,
                    return_intermediate_maps=True
                )
                
                result["localization"] = {
                    "probability": localization_result["probability"],
                    "num_regions": len(localization_result["bboxes"]),
                    "bboxes": localization_result["bboxes"],
                    "has_tampering": localization_result["probability"] > 0.5
                }
                
                result["visualizations"] = {
                    "heatmap": localization_result.get("heatmap"),
                    "mask": localization_result.get("mask"),
                    "fused_map": localization_result.get("fused_map")
                }
                
                if "intermediate_maps" in localization_result:
                    result["forensic_analysis"] = {
                        name: map_data.mean() 
                        for name, map_data in localization_result["intermediate_maps"].items()
                    }
                
                print(f"\n✅ Localization completed!")
                print(f"   - Tampering probability: {result['localization']['probability']:.2%}")
                print(f"   - Detected regions: {result['localization']['num_regions']}")
                
            except Exception as e:
                print(f"❌ Localization failed: {e}")
                result["localization"] = {"error": str(e)}
        
        # Generate risk assessment
        result["risk_assessment"] = self._assess_risk(result)
        
        return result
    
    def _basic_prediction(self, image_path: str) -> Dict[str, Any]:
        """
        Run basic forgery detection (binary classification).
        Uses your existing predict function if available.
        """
        try:
            # Try to use existing predict function
            prediction = predict_image(image_path, self.model, self.device)
            return prediction
        except:
            # Fallback: simple forward pass
            from torchvision import transforms
            from PIL import Image
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                
                if output.dim() == 1:
                    prob = torch.sigmoid(output).item()
                else:
                    prob = torch.softmax(output, dim=1)[0, 1].item()
                
                is_forged = prob > 0.5
                
                return {
                    "class": "forged" if is_forged else "authentic",
                    "confidence": prob if is_forged else (1 - prob),
                    "forgery_score": prob
                }
    
    def _assess_risk(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive risk assessment.
        """
        assessment = {
            "risk_level": "unknown",
            "recommendation": "Further investigation needed",
            "severity_score": 0.0
        }
        
        # Calculate severity
        basic_conf = result.get("basic_detection", {}).get("confidence", 0.5)
        loc_prob = result.get("localization", {}).get("probability", 0.5)
        num_regions = result.get("localization", {}).get("num_regions", 0)
        
        # Weighted severity score
        severity = (basic_conf * 0.4 + loc_prob * 0.6)
        assessment["severity_score"] = severity
        
        # Risk level classification
        if severity < 0.3:
            assessment["risk_level"] = "low"
            assessment["recommendation"] = "✅ Document appears authentic"
        elif severity < 0.5:
            assessment["risk_level"] = "medium"
            assessment["recommendation"] = "⚠️ Manual review recommended"
        elif severity < 0.7:
            assessment["risk_level"] = "high"
            assessment["recommendation"] = "🚨 Document likely tampered - flag for investigation"
        else:
            assessment["risk_level"] = "critical"
            assessment["recommendation"] = "🚫 Reject document - high probability of forgery"
        
        # Add region-based context
        if num_regions > 0:
            assessment["details"] = f"Detected {num_regions} suspicious region(s)"
        
        return assessment
    
    def batch_analyze(self, image_paths: list, **kwargs) -> list:
        """
        Analyze multiple documents.
        
        Args:
            image_paths: List of image paths
            **kwargs: Additional arguments for analyze_document
        
        Returns:
            List of analysis results
        """
        results = []
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {img_path}")
            
            try:
                result = self.analyze_document(img_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                results.append({
                    "image_path": img_path,
                    "error": str(e)
                })
        
        return results
    
    def generate_report(self, result: Dict[str, Any], output_file: str = None):
        """
        Generate a detailed analysis report.
        
        Args:
            result: Analysis result from analyze_document
            output_file: Optional file to save report
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("DOCUMENT FORENSIC ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Basic info
        report_lines.append(f"Document: {os.path.basename(result['image_path'])}")
        report_lines.append(f"Analysis Date: {Path(result['image_path']).stat().st_mtime}")
        report_lines.append("")
        
        # Basic detection
        basic = result.get("basic_detection", {})
        report_lines.append("--- BINARY CLASSIFICATION ---")
        report_lines.append(f"Classification: {basic.get('class', 'N/A').upper()}")
        report_lines.append(f"Confidence: {basic.get('confidence', 0):.2%}")
        report_lines.append("")
        
        # Localization
        if "localization" in result and "error" not in result["localization"]:
            loc = result["localization"]
            report_lines.append("--- TAMPERING LOCALIZATION ---")
            report_lines.append(f"Tampering Probability: {loc['probability']:.2%}")
            report_lines.append(f"Detected Regions: {loc['num_regions']}")
            
            if loc.get("bboxes"):
                report_lines.append("\nSuspicious Regions:")
                for i, (x, y, w, h) in enumerate(loc["bboxes"], 1):
                    report_lines.append(f"  Region {i}: ({x}, {y}) - {w}×{h} pixels")
            report_lines.append("")
        
        # Forensic analysis
        if "forensic_analysis" in result:
            report_lines.append("--- FORENSIC TECHNIQUES ---")
            for tech, score in result["forensic_analysis"].items():
                report_lines.append(f"{tech.upper()}: {score:.3f}")
            report_lines.append("")
        
        # Risk assessment
        risk = result.get("risk_assessment", {})
        report_lines.append("--- RISK ASSESSMENT ---")
        report_lines.append(f"Risk Level: {risk.get('risk_level', 'unknown').upper()}")
        report_lines.append(f"Severity Score: {risk.get('severity_score', 0):.2%}")
        report_lines.append(f"Recommendation: {risk.get('recommendation', 'N/A')}")
        report_lines.append("")
        
        report_lines.append("=" * 70)
        
        report_text = "\n".join(report_lines)
        
        # Print to console
        print(report_text)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\n✓ Report saved to: {output_file}")
        
        return report_text


def main():
    """
    Example usage of the integrated system.
    """
    
    print("\n" + "="*70)
    print("DocuForge - Enhanced Document Analysis with Localization")
    print("="*70 + "\n")
    
    # Initialize analyzer
    try:
        analyzer = EnhancedDocumentAnalyzer(
            model_path="saved_models/best_model.pth"
        )
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        return
    
    # Example 1: Single document analysis
    test_image = "dataset/test/forged/sample_document.jpg"
    
    if os.path.exists(test_image):
        print("📄 Example 1: Single Document Analysis\n")
        
        result = analyzer.analyze_document(
            image_path=test_image,
            enable_localization=True,
            sensitivity=0.5,
            save_results=True
        )
        
        # Generate report
        analyzer.generate_report(
            result,
            output_file="tampering_results/analysis_report.txt"
        )
    else:
        print(f"⚠️  Test image not found: {test_image}")
        print("Please provide a valid test image path.")
    
    # Example 2: Batch analysis
    print("\n\n📚 Example 2: Batch Document Analysis\n")
    
    test_dir = Path("dataset/test/forged")
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg"))[:3]  # First 3 images
        
        if test_images:
            batch_results = analyzer.batch_analyze(
                [str(p) for p in test_images],
                enable_localization=True,
                save_results=True
            )
            
            # Summary
            print("\n\n📊 Batch Analysis Summary:")
            print("-" * 70)
            for i, result in enumerate(batch_results, 1):
                if "error" not in result:
                    risk = result["risk_assessment"]
                    loc = result.get("localization", {})
                    print(f"{i}. {os.path.basename(result['image_path'])}")
                    print(f"   Risk: {risk['risk_level'].upper()} "
                          f"(Score: {risk['severity_score']:.2%})")
                    print(f"   Regions: {loc.get('num_regions', 0)}")
                    print()


if __name__ == "__main__":
    main()
