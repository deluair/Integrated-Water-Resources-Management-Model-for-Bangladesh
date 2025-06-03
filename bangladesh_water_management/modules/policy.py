"""Policy Analysis Module for Bangladesh Water Management.

This module handles policy evaluation, regulatory framework analysis,
governance assessment, and institutional capacity modeling for water management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from scipy.optimize import minimize
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class PolicyAnalyzer:
    """Analyzes water management policies and governance frameworks.
    
    This class implements policy evaluation methods, institutional analysis,
    regulatory impact assessment, and governance optimization for water sector.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize policy analyzer.
        
        Args:
            config: Configuration dictionary containing policy parameters
        """
        self.config = config
        self.policy_config = config.get('policy', {})
        
        # Initialize policy frameworks
        self.policy_frameworks = self._initialize_policy_frameworks()
        
        # Initialize institutional structures
        self.institutional_structures = self._initialize_institutional_structures()
        
        # Initialize governance indicators
        self.governance_indicators = self._initialize_governance_indicators()
        
        # Initialize policy instruments
        self.policy_instruments = self._initialize_policy_instruments()
        
        logger.info("Policy Analyzer initialized")
    
    def _initialize_policy_frameworks(self) -> Dict[str, Any]:
        """Initialize water policy frameworks for Bangladesh."""
        return {
            'national_policies': {
                'national_water_policy_1999': {
                    'status': 'Active',
                    'last_updated': 1999,
                    'key_principles': [
                        'Water as a basic human need',
                        'Integrated water resources management',
                        'Participatory approach',
                        'Environmental sustainability',
                        'Economic efficiency'
                    ],
                    'implementation_score': 6.5,  # Out of 10
                    'effectiveness_rating': 'Moderate'
                },
                'national_water_act_2013': {
                    'status': 'Active',
                    'last_updated': 2013,
                    'key_provisions': [
                        'Water rights allocation',
                        'Water use licensing',
                        'Pollution control',
                        'Institutional arrangements',
                        'Dispute resolution'
                    ],
                    'implementation_score': 5.8,
                    'effectiveness_rating': 'Moderate'
                },
                'delta_plan_2100': {
                    'status': 'Active',
                    'last_updated': 2018,
                    'planning_horizon': 2100,
                    'investment_requirement_billion_usd': 37,
                    'key_strategies': [
                        'Adaptive delta management',
                        'Climate resilience',
                        'Sustainable development',
                        'Disaster risk reduction',
                        'Ecosystem-based adaptation'
                    ],
                    'implementation_score': 4.2,
                    'effectiveness_rating': 'Early Stage'
                }
            },
            'sectoral_policies': {
                'urban_water_supply': {
                    'service_coverage_target_2030': 0.95,
                    'quality_standards': 'WHO Guidelines',
                    'tariff_policy': 'Cost Recovery',
                    'private_sector_participation': 'Limited',
                    'implementation_challenges': [
                        'Institutional capacity',
                        'Financial sustainability',
                        'Technical expertise',
                        'Coordination issues'
                    ]
                },
                'rural_water_supply': {
                    'service_coverage_target_2030': 0.85,
                    'technology_preference': 'Community-based',
                    'financing_mechanism': 'Government + Development Partners',
                    'sustainability_concerns': [
                        'Operation and maintenance',
                        'Community ownership',
                        'Technical support',
                        'Financial management'
                    ]
                },
                'irrigation_policy': {
                    'efficiency_target': 0.65,  # 65% irrigation efficiency
                    'modernization_priority': 'High',
                    'water_pricing': 'Subsidized',
                    'technology_promotion': [
                        'Drip irrigation',
                        'Sprinkler systems',
                        'Precision agriculture',
                        'Water-efficient crops'
                    ]
                },
                'flood_management': {
                    'approach': 'Integrated Flood Management',
                    'early_warning_coverage': 0.70,
                    'structural_measures': 'Embankments + Polders',
                    'non_structural_measures': [
                        'Land use planning',
                        'Building codes',
                        'Insurance schemes',
                        'Community preparedness'
                    ]
                }
            },
            'international_frameworks': {
                'sdg_6': {
                    'targets': {
                        '6.1': 'Universal access to safe drinking water',
                        '6.2': 'Universal access to sanitation and hygiene',
                        '6.3': 'Improve water quality and wastewater treatment',
                        '6.4': 'Increase water-use efficiency',
                        '6.5': 'Implement integrated water resources management',
                        '6.6': 'Protect water-related ecosystems'
                    },
                    'progress_score': 5.5,  # Out of 10
                    'achievement_likelihood_2030': 'Moderate'
                },
                'paris_agreement': {
                    'water_related_commitments': [
                        'Climate adaptation in water sector',
                        'Resilient water infrastructure',
                        'Ecosystem-based adaptation'
                    ],
                    'implementation_status': 'In Progress'
                },
                'sendai_framework': {
                    'disaster_risk_reduction_priorities': [
                        'Understanding disaster risk',
                        'Strengthening governance',
                        'Investing in resilience',
                        'Enhancing preparedness'
                    ],
                    'water_sector_integration': 'Moderate'
                }
            }
        }
    
    def _initialize_institutional_structures(self) -> Dict[str, Any]:
        """Initialize institutional structures for water governance."""
        return {
            'national_level': {
                'ministry_of_water_resources': {
                    'role': 'Policy formulation and coordination',
                    'capacity_score': 6.8,
                    'key_functions': [
                        'Policy development',
                        'Inter-ministerial coordination',
                        'International cooperation',
                        'Sector monitoring'
                    ],
                    'capacity_constraints': [
                        'Limited technical expertise',
                        'Inadequate coordination mechanisms',
                        'Weak monitoring systems'
                    ]
                },
                'bangladesh_water_development_board': {
                    'role': 'Water resources development and management',
                    'capacity_score': 7.2,
                    'key_functions': [
                        'Flood control and drainage',
                        'Irrigation development',
                        'River management',
                        'Coastal protection'
                    ],
                    'strengths': [
                        'Technical expertise',
                        'Implementation experience',
                        'Field presence'
                    ],
                    'weaknesses': [
                        'Limited financial resources',
                        'Aging infrastructure',
                        'Coordination challenges'
                    ]
                },
                'department_of_public_health_engineering': {
                    'role': 'Water supply and sanitation',
                    'capacity_score': 6.5,
                    'key_functions': [
                        'Rural water supply',
                        'Sanitation programs',
                        'Hygiene promotion',
                        'Quality monitoring'
                    ],
                    'coverage': {
                        'rural_water_supply': 0.85,
                        'rural_sanitation': 0.65,
                        'hygiene_awareness': 0.70
                    }
                },
                'water_and_sewerage_authorities': {
                    'role': 'Urban water and sewerage services',
                    'number_of_authorities': 12,
                    'average_capacity_score': 5.8,
                    'service_coverage': {
                        'water_supply': 0.75,
                        'sewerage': 0.25,
                        'drainage': 0.55
                    },
                    'financial_sustainability': 'Poor',
                    'common_challenges': [
                        'Revenue collection',
                        'System losses',
                        'Maintenance backlogs',
                        'Staff capacity'
                    ]
                }
            },
            'local_level': {
                'union_parishads': {
                    'role': 'Local water and sanitation services',
                    'number': 4554,
                    'capacity_score': 4.2,
                    'functions': [
                        'Community water points',
                        'Sanitation facilities',
                        'Local drainage',
                        'Hygiene promotion'
                    ],
                    'capacity_building_needs': [
                        'Technical skills',
                        'Financial management',
                        'Project planning',
                        'Community mobilization'
                    ]
                },
                'water_management_associations': {
                    'role': 'Participatory water management',
                    'number': 1200,
                    'effectiveness_score': 5.5,
                    'activities': [
                        'Irrigation management',
                        'Water user coordination',
                        'Conflict resolution',
                        'System maintenance'
                    ],
                    'success_factors': [
                        'Strong leadership',
                        'Clear water rights',
                        'Adequate financing',
                        'Technical support'
                    ]
                }
            },
            'coordination_mechanisms': {
                'national_water_resources_council': {
                    'status': 'Established but inactive',
                    'intended_role': 'High-level policy coordination',
                    'effectiveness': 'Very Low',
                    'revival_priority': 'High'
                },
                'inter_ministerial_committees': {
                    'number': 8,
                    'average_effectiveness': 'Low',
                    'common_issues': [
                        'Irregular meetings',
                        'Lack of decision-making authority',
                        'Poor follow-up',
                        'Conflicting mandates'
                    ]
                },
                'river_basin_organizations': {
                    'status': 'Proposed but not established',
                    'potential_benefits': [
                        'Integrated basin management',
                        'Stakeholder coordination',
                        'Conflict resolution',
                        'Resource optimization'
                    ],
                    'establishment_priority': 'High'
                }
            }
        }
    
    def _initialize_governance_indicators(self) -> Dict[str, Any]:
        """Initialize water governance indicators and benchmarks."""
        return {
            'oecd_water_governance_principles': {
                'effectiveness': {
                    'score': 5.2,  # Out of 10
                    'indicators': {
                        'clear_roles_responsibilities': 4.8,
                        'appropriate_scales': 5.5,
                        'policy_coherence': 4.9,
                        'capacity_building': 5.8
                    }
                },
                'efficiency': {
                    'score': 4.8,
                    'indicators': {
                        'financing_sustainability': 4.2,
                        'regulatory_frameworks': 5.1,
                        'innovative_governance': 4.5,
                        'data_information': 5.4
                    }
                },
                'trust_engagement': {
                    'score': 5.5,
                    'indicators': {
                        'stakeholder_engagement': 6.2,
                        'transparency_accountability': 4.8,
                        'integrity_practices': 5.1,
                        'public_participation': 6.0
                    }
                }
            },
            'world_bank_governance_indicators': {
                'voice_accountability': 3.8,  # Out of 10
                'political_stability': 4.2,
                'government_effectiveness': 4.5,
                'regulatory_quality': 4.1,
                'rule_of_law': 3.9,
                'control_of_corruption': 2.8
            },
            'water_specific_indicators': {
                'institutional_capacity': {
                    'policy_formulation': 6.0,
                    'implementation_capacity': 5.5,
                    'monitoring_evaluation': 4.8,
                    'coordination_mechanisms': 4.2,
                    'technical_expertise': 6.5
                },
                'financial_governance': {
                    'budget_allocation': 5.5,
                    'expenditure_efficiency': 4.8,
                    'cost_recovery': 3.5,
                    'tariff_setting': 4.2,
                    'subsidy_targeting': 3.8
                },
                'regulatory_framework': {
                    'legal_framework_completeness': 6.8,
                    'enforcement_capacity': 4.5,
                    'compliance_monitoring': 4.2,
                    'dispute_resolution': 5.0,
                    'standards_setting': 6.2
                },
                'participation_transparency': {
                    'stakeholder_consultation': 5.8,
                    'information_disclosure': 4.5,
                    'grievance_mechanisms': 4.8,
                    'community_participation': 6.5,
                    'civil_society_engagement': 6.0
                }
            }
        }
    
    def _initialize_policy_instruments(self) -> Dict[str, Any]:
        """Initialize available policy instruments and their characteristics."""
        return {
            'regulatory_instruments': {
                'water_use_permits': {
                    'description': 'Licensing system for water abstraction',
                    'current_status': 'Partially implemented',
                    'effectiveness_score': 4.5,
                    'coverage': 0.35,  # 35% of water users
                    'enforcement_capacity': 'Weak',
                    'improvement_potential': 'High'
                },
                'pollution_standards': {
                    'description': 'Effluent discharge standards',
                    'current_status': 'Established but poorly enforced',
                    'effectiveness_score': 3.8,
                    'compliance_rate': 0.25,
                    'monitoring_capacity': 'Limited',
                    'penalties': 'Inadequate'
                },
                'building_codes': {
                    'description': 'Water-related building standards',
                    'current_status': 'Exists but outdated',
                    'effectiveness_score': 4.2,
                    'climate_resilience': 'Limited',
                    'enforcement': 'Inconsistent'
                },
                'land_use_regulations': {
                    'description': 'Water resource protection zoning',
                    'current_status': 'Weak implementation',
                    'effectiveness_score': 3.5,
                    'coverage': 0.20,
                    'enforcement_challenges': [
                        'Lack of coordination',
                        'Conflicting interests',
                        'Limited capacity'
                    ]
                }
            },
            'economic_instruments': {
                'water_pricing': {
                    'description': 'Volumetric water pricing',
                    'current_status': 'Limited implementation',
                    'effectiveness_score': 4.0,
                    'sectors_covered': ['Urban water supply', 'Some irrigation'],
                    'cost_recovery_rate': 0.65,
                    'affordability_concerns': 'Moderate'
                },
                'pollution_charges': {
                    'description': 'Fees for wastewater discharge',
                    'current_status': 'Not implemented',
                    'potential_effectiveness': 6.5,
                    'implementation_barriers': [
                        'Lack of monitoring',
                        'Administrative capacity',
                        'Political resistance'
                    ]
                },
                'subsidies_incentives': {
                    'description': 'Financial incentives for water efficiency',
                    'current_status': 'Limited scope',
                    'effectiveness_score': 5.2,
                    'target_areas': [
                        'Irrigation modernization',
                        'Rainwater harvesting',
                        'Water treatment'
                    ],
                    'budget_allocation': 'Insufficient'
                },
                'payment_ecosystem_services': {
                    'description': 'Payments for watershed services',
                    'current_status': 'Pilot projects only',
                    'potential_effectiveness': 7.0,
                    'implementation_requirements': [
                        'Legal framework',
                        'Institutional capacity',
                        'Financing mechanisms'
                    ]
                }
            },
            'information_instruments': {
                'public_awareness_campaigns': {
                    'description': 'Water conservation awareness',
                    'current_status': 'Ongoing but limited',
                    'effectiveness_score': 5.8,
                    'reach': 0.45,  # 45% of population
                    'behavior_change_impact': 'Moderate'
                },
                'water_quality_reporting': {
                    'description': 'Public disclosure of water quality',
                    'current_status': 'Very limited',
                    'effectiveness_score': 3.2,
                    'coverage': 0.15,
                    'improvement_needs': [
                        'Monitoring network',
                        'Data management',
                        'Communication systems'
                    ]
                },
                'early_warning_systems': {
                    'description': 'Flood and drought warnings',
                    'current_status': 'Partially operational',
                    'effectiveness_score': 6.5,
                    'coverage': 0.70,
                    'accuracy': 'Good for floods, poor for droughts'
                }
            },
            'participatory_instruments': {
                'community_based_management': {
                    'description': 'Community-led water management',
                    'current_status': 'Widespread in rural areas',
                    'effectiveness_score': 6.8,
                    'sustainability_rate': 0.65,
                    'success_factors': [
                        'Strong community ownership',
                        'Technical support',
                        'Financial sustainability'
                    ]
                },
                'stakeholder_platforms': {
                    'description': 'Multi-stakeholder coordination',
                    'current_status': 'Limited and ad-hoc',
                    'effectiveness_score': 4.5,
                    'institutionalization': 'Weak',
                    'potential_benefits': [
                        'Improved coordination',
                        'Conflict resolution',
                        'Knowledge sharing'
                    ]
                },
                'public_private_partnerships': {
                    'description': 'PPP in water services',
                    'current_status': 'Very limited',
                    'effectiveness_score': 5.0,
                    'barriers': [
                        'Regulatory uncertainty',
                        'Political risks',
                        'Limited private sector interest'
                    ],
                    'potential_sectors': [
                        'Urban water supply',
                        'Wastewater treatment',
                        'Irrigation modernization'
                    ]
                }
            }
        }
    
    def evaluate_policy_effectiveness(self,
                                   policy_name: str,
                                   evaluation_criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate the effectiveness of a specific water policy.
        
        Args:
            policy_name: Name of the policy to evaluate
            evaluation_criteria: Specific criteria for evaluation
            
        Returns:
            Comprehensive policy effectiveness assessment
        """
        if evaluation_criteria is None:
            evaluation_criteria = [
                'implementation_progress',
                'outcome_achievement',
                'stakeholder_satisfaction',
                'cost_effectiveness',
                'sustainability'
            ]
        
        # Find policy in frameworks
        policy_data = self._find_policy_data(policy_name)
        
        if not policy_data:
            raise ValueError(f"Policy '{policy_name}' not found in policy frameworks")
        
        # Conduct evaluation
        evaluation_results = {}
        
        for criterion in evaluation_criteria:
            evaluation_results[criterion] = self._evaluate_policy_criterion(
                policy_data, criterion
            )
        
        # Overall assessment
        overall_score = np.mean([result['score'] for result in evaluation_results.values()])
        
        # Generate recommendations
        recommendations = self._generate_policy_recommendations(
            policy_name, policy_data, evaluation_results
        )
        
        return {
            'policy_name': policy_name,
            'evaluation_criteria': evaluation_criteria,
            'detailed_results': evaluation_results,
            'overall_score': overall_score,
            'overall_rating': self._score_to_rating(overall_score),
            'recommendations': recommendations,
            'priority_actions': self._identify_priority_actions(evaluation_results)
        }
    
    def _find_policy_data(self, policy_name: str) -> Optional[Dict]:
        """Find policy data in the frameworks."""
        # Search in national policies
        for policy_key, policy_data in self.policy_frameworks['national_policies'].items():
            if policy_name.lower() in policy_key.lower():
                return policy_data
        
        # Search in sectoral policies
        for sector_key, sector_data in self.policy_frameworks['sectoral_policies'].items():
            if policy_name.lower() in sector_key.lower():
                return sector_data
        
        return None
    
    def _evaluate_policy_criterion(self, policy_data: Dict, criterion: str) -> Dict[str, Any]:
        """Evaluate a specific criterion for a policy."""
        if criterion == 'implementation_progress':
            score = policy_data.get('implementation_score', 5.0)
            return {
                'score': score,
                'rating': self._score_to_rating(score),
                'evidence': [
                    f"Implementation score: {score}/10",
                    f"Status: {policy_data.get('status', 'Unknown')}"
                ],
                'challenges': policy_data.get('implementation_challenges', [])
            }
        
        elif criterion == 'outcome_achievement':
            # Assess based on targets vs achievements
            if 'service_coverage_target_2030' in policy_data:
                target = policy_data['service_coverage_target_2030']
                current = 0.75  # Assumed current coverage
                progress = (current / target) * 10
                score = min(10, progress)
            else:
                score = policy_data.get('effectiveness_rating', 'Moderate')
                score = {'Excellent': 9, 'Good': 7, 'Moderate': 5, 'Poor': 3, 'Very Poor': 1}.get(score, 5)
            
            return {
                'score': score,
                'rating': self._score_to_rating(score),
                'evidence': [f"Outcome achievement score: {score}/10"],
                'gaps': self._identify_outcome_gaps(policy_data)
            }
        
        elif criterion == 'stakeholder_satisfaction':
            # Simplified stakeholder satisfaction assessment
            participation_score = self.governance_indicators['water_specific_indicators']['participation_transparency']
            avg_satisfaction = np.mean(list(participation_score.values()))
            
            return {
                'score': avg_satisfaction,
                'rating': self._score_to_rating(avg_satisfaction),
                'evidence': [f"Average stakeholder satisfaction: {avg_satisfaction}/10"],
                'key_concerns': [
                    'Limited consultation processes',
                    'Inadequate feedback mechanisms',
                    'Unequal representation'
                ]
            }
        
        elif criterion == 'cost_effectiveness':
            # Assess cost-effectiveness based on available data
            if 'investment_requirement_billion_usd' in policy_data:
                investment = policy_data['investment_requirement_billion_usd']
                # Simplified cost-effectiveness calculation
                score = max(1, 10 - (investment / 10))  # Inverse relationship with cost
            else:
                score = 6.0  # Default moderate score
            
            return {
                'score': score,
                'rating': self._score_to_rating(score),
                'evidence': [f"Cost-effectiveness score: {score}/10"],
                'cost_concerns': [
                    'High implementation costs',
                    'Limited budget allocation',
                    'Inefficient resource use'
                ]
            }
        
        elif criterion == 'sustainability':
            # Assess long-term sustainability
            sustainability_factors = {
                'financial_sustainability': 4.5,
                'institutional_sustainability': 5.2,
                'environmental_sustainability': 6.0,
                'social_sustainability': 5.8
            }
            
            score = np.mean(list(sustainability_factors.values()))
            
            return {
                'score': score,
                'rating': self._score_to_rating(score),
                'evidence': [f"Sustainability score: {score}/10"],
                'sustainability_factors': sustainability_factors,
                'risks': [
                    'Financial constraints',
                    'Institutional capacity gaps',
                    'Climate change impacts'
                ]
            }
        
        else:
            # Default evaluation for unknown criteria
            return {
                'score': 5.0,
                'rating': 'Moderate',
                'evidence': [f"Default assessment for {criterion}"],
                'note': f"Specific evaluation method for {criterion} not implemented"
            }
    
    def _score_to_rating(self, score: float) -> str:
        """Convert numerical score to rating."""
        if score >= 8.5:
            return 'Excellent'
        elif score >= 7.0:
            return 'Good'
        elif score >= 5.5:
            return 'Moderate'
        elif score >= 3.5:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _identify_outcome_gaps(self, policy_data: Dict) -> List[str]:
        """Identify gaps in policy outcomes."""
        gaps = []
        
        if 'service_coverage_target_2030' in policy_data:
            target = policy_data['service_coverage_target_2030']
            if target > 0.8:
                gaps.append("Ambitious coverage targets may be difficult to achieve")
        
        if 'implementation_challenges' in policy_data:
            gaps.extend([f"Challenge: {challenge}" for challenge in policy_data['implementation_challenges']])
        
        return gaps
    
    def _generate_policy_recommendations(self,
                                       policy_name: str,
                                       policy_data: Dict,
                                       evaluation_results: Dict) -> List[str]:
        """Generate recommendations for policy improvement."""
        recommendations = []
        
        # Based on overall performance
        overall_score = np.mean([result['score'] for result in evaluation_results.values()])
        
        if overall_score < 5.0:
            recommendations.append("Consider major policy revision or replacement")
        elif overall_score < 7.0:
            recommendations.append("Implement targeted improvements to address key weaknesses")
        else:
            recommendations.append("Focus on fine-tuning and scaling successful elements")
        
        # Specific recommendations based on criteria
        for criterion, result in evaluation_results.items():
            if result['score'] < 5.0:
                if criterion == 'implementation_progress':
                    recommendations.append("Strengthen implementation mechanisms and capacity")
                elif criterion == 'stakeholder_satisfaction':
                    recommendations.append("Enhance stakeholder engagement and consultation processes")
                elif criterion == 'cost_effectiveness':
                    recommendations.append("Review and optimize resource allocation and efficiency")
                elif criterion == 'sustainability':
                    recommendations.append("Develop long-term sustainability strategies and financing")
        
        # Policy-specific recommendations
        if 'delta_plan' in policy_name.lower():
            recommendations.extend([
                "Accelerate implementation of priority projects",
                "Strengthen institutional coordination mechanisms",
                "Enhance climate risk assessment and adaptation measures"
            ])
        
        return recommendations
    
    def _identify_priority_actions(self, evaluation_results: Dict) -> List[Dict[str, Any]]:
        """Identify priority actions based on evaluation results."""
        actions = []
        
        # Sort criteria by score (lowest first)
        sorted_criteria = sorted(
            evaluation_results.items(),
            key=lambda x: x[1]['score']
        )
        
        # Generate actions for lowest-scoring criteria
        for criterion, result in sorted_criteria[:3]:  # Top 3 priorities
            if result['score'] < 7.0:
                action = {
                    'area': criterion,
                    'priority': 'High' if result['score'] < 4.0 else 'Medium',
                    'timeframe': 'Short-term (1-2 years)' if result['score'] < 4.0 else 'Medium-term (2-5 years)',
                    'description': self._generate_action_description(criterion, result),
                    'estimated_cost': self._estimate_action_cost(criterion),
                    'expected_impact': 'High' if result['score'] < 4.0 else 'Medium'
                }
                actions.append(action)
        
        return actions
    
    def _generate_action_description(self, criterion: str, result: Dict) -> str:
        """Generate action description for a criterion."""
        descriptions = {
            'implementation_progress': "Develop detailed implementation roadmap with clear milestones and accountability mechanisms",
            'outcome_achievement': "Establish robust monitoring and evaluation system to track progress against targets",
            'stakeholder_satisfaction': "Create formal stakeholder engagement framework with regular consultation processes",
            'cost_effectiveness': "Conduct comprehensive cost-benefit analysis and optimize resource allocation",
            'sustainability': "Develop long-term financing strategy and institutional capacity building program"
        }
        
        return descriptions.get(criterion, f"Address key issues in {criterion}")
    
    def _estimate_action_cost(self, criterion: str) -> str:
        """Estimate cost for implementing action."""
        cost_estimates = {
            'implementation_progress': 'Medium ($500K - $2M)',
            'outcome_achievement': 'Low ($100K - $500K)',
            'stakeholder_satisfaction': 'Low ($50K - $200K)',
            'cost_effectiveness': 'Low ($100K - $300K)',
            'sustainability': 'High ($2M - $10M)'
        }
        
        return cost_estimates.get(criterion, 'Medium ($500K - $2M)')
    
    def analyze_institutional_capacity(self,
                                     institution_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze institutional capacity for water governance.
        
        Args:
            institution_name: Specific institution to analyze (optional)
            
        Returns:
            Institutional capacity analysis
        """
        if institution_name:
            # Analyze specific institution
            institution_data = self._find_institution_data(institution_name)
            if not institution_data:
                raise ValueError(f"Institution '{institution_name}' not found")
            
            return self._analyze_single_institution(institution_name, institution_data)
        else:
            # Analyze overall institutional landscape
            return self._analyze_institutional_landscape()
    
    def _find_institution_data(self, institution_name: str) -> Optional[Dict]:
        """Find institution data in the structures."""
        # Search in national level institutions
        for inst_key, inst_data in self.institutional_structures['national_level'].items():
            if institution_name.lower() in inst_key.lower():
                return inst_data
        
        # Search in local level institutions
        for inst_key, inst_data in self.institutional_structures['local_level'].items():
            if institution_name.lower() in inst_key.lower():
                return inst_data
        
        return None
    
    def _analyze_single_institution(self, name: str, data: Dict) -> Dict[str, Any]:
        """Analyze capacity of a single institution."""
        capacity_score = data.get('capacity_score', 5.0)
        
        # Capacity assessment dimensions
        capacity_dimensions = {
            'technical_capacity': self._assess_technical_capacity(data),
            'financial_capacity': self._assess_financial_capacity(data),
            'human_resources': self._assess_human_resources(data),
            'organizational_systems': self._assess_organizational_systems(data),
            'leadership_governance': self._assess_leadership_governance(data)
        }
        
        # Identify strengths and weaknesses
        strengths = data.get('strengths', [])
        weaknesses = data.get('weaknesses', [])
        capacity_constraints = data.get('capacity_constraints', [])
        
        # Development recommendations
        recommendations = self._generate_capacity_recommendations(
            name, capacity_dimensions, strengths, weaknesses
        )
        
        return {
            'institution_name': name,
            'overall_capacity_score': capacity_score,
            'capacity_rating': self._score_to_rating(capacity_score),
            'capacity_dimensions': capacity_dimensions,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'capacity_constraints': capacity_constraints,
            'development_recommendations': recommendations,
            'priority_interventions': self._identify_capacity_priorities(capacity_dimensions)
        }
    
    def _assess_technical_capacity(self, data: Dict) -> Dict[str, Any]:
        """Assess technical capacity of institution."""
        # Simplified technical capacity assessment
        base_score = data.get('capacity_score', 5.0)
        
        # Adjust based on functions and expertise
        if 'technical expertise' in data.get('strengths', []):
            technical_score = min(10, base_score + 1.5)
        elif 'Limited technical expertise' in data.get('capacity_constraints', []):
            technical_score = max(1, base_score - 1.5)
        else:
            technical_score = base_score
        
        return {
            'score': technical_score,
            'rating': self._score_to_rating(technical_score),
            'key_areas': data.get('key_functions', []),
            'gaps': [constraint for constraint in data.get('capacity_constraints', []) 
                    if 'technical' in constraint.lower()]
        }
    
    def _assess_financial_capacity(self, data: Dict) -> Dict[str, Any]:
        """Assess financial capacity of institution."""
        base_score = data.get('capacity_score', 5.0)
        
        # Adjust based on financial indicators
        if 'Limited financial resources' in data.get('weaknesses', []):
            financial_score = max(1, base_score - 2.0)
        elif data.get('financial_sustainability') == 'Poor':
            financial_score = max(1, base_score - 2.5)
        else:
            financial_score = base_score
        
        return {
            'score': financial_score,
            'rating': self._score_to_rating(financial_score),
            'sustainability': data.get('financial_sustainability', 'Unknown'),
            'challenges': [challenge for challenge in data.get('common_challenges', []) 
                          if any(word in challenge.lower() for word in ['revenue', 'financial', 'cost'])]
        }
    
    def _assess_human_resources(self, data: Dict) -> Dict[str, Any]:
        """Assess human resource capacity."""
        base_score = data.get('capacity_score', 5.0)
        
        # Adjust based on HR indicators
        if 'Staff capacity' in data.get('common_challenges', []):
            hr_score = max(1, base_score - 1.0)
        else:
            hr_score = base_score
        
        return {
            'score': hr_score,
            'rating': self._score_to_rating(hr_score),
            'capacity_building_needs': data.get('capacity_building_needs', []),
            'challenges': [challenge for challenge in data.get('common_challenges', []) 
                          if 'staff' in challenge.lower()]
        }
    
    def _assess_organizational_systems(self, data: Dict) -> Dict[str, Any]:
        """Assess organizational systems and processes."""
        base_score = data.get('capacity_score', 5.0)
        
        # Adjust based on system indicators
        if 'Weak monitoring systems' in data.get('capacity_constraints', []):
            systems_score = max(1, base_score - 1.5)
        elif 'Implementation experience' in data.get('strengths', []):
            systems_score = min(10, base_score + 1.0)
        else:
            systems_score = base_score
        
        return {
            'score': systems_score,
            'rating': self._score_to_rating(systems_score),
            'system_strengths': [s for s in data.get('strengths', []) if 'system' in s.lower()],
            'system_weaknesses': [w for w in data.get('weaknesses', []) if 'system' in w.lower()]
        }
    
    def _assess_leadership_governance(self, data: Dict) -> Dict[str, Any]:
        """Assess leadership and governance capacity."""
        base_score = data.get('capacity_score', 5.0)
        
        # Adjust based on governance indicators
        if 'Coordination challenges' in data.get('weaknesses', []):
            governance_score = max(1, base_score - 1.0)
        else:
            governance_score = base_score
        
        return {
            'score': governance_score,
            'rating': self._score_to_rating(governance_score),
            'governance_challenges': [w for w in data.get('weaknesses', []) 
                                    if any(word in w.lower() for word in ['coordination', 'governance', 'leadership'])]
        }
    
    def _generate_capacity_recommendations(self,
                                        name: str,
                                        capacity_dimensions: Dict,
                                        strengths: List[str],
                                        weaknesses: List[str]) -> List[str]:
        """Generate capacity development recommendations."""
        recommendations = []
        
        # Based on lowest-scoring dimensions
        sorted_dimensions = sorted(
            capacity_dimensions.items(),
            key=lambda x: x[1]['score']
        )
        
        for dimension, assessment in sorted_dimensions[:2]:  # Top 2 priorities
            if assessment['score'] < 6.0:
                if dimension == 'technical_capacity':
                    recommendations.append("Invest in technical training and skill development programs")
                elif dimension == 'financial_capacity':
                    recommendations.append("Develop sustainable financing mechanisms and improve revenue generation")
                elif dimension == 'human_resources':
                    recommendations.append("Implement comprehensive human resource development strategy")
                elif dimension == 'organizational_systems':
                    recommendations.append("Strengthen organizational systems and processes")
                elif dimension == 'leadership_governance':
                    recommendations.append("Enhance leadership capacity and governance structures")
        
        # Leverage strengths
        if strengths:
            recommendations.append(f"Build on existing strengths: {', '.join(strengths[:2])}")
        
        # Address critical weaknesses
        if weaknesses:
            recommendations.append(f"Address critical weaknesses: {', '.join(weaknesses[:2])}")
        
        return recommendations
    
    def _identify_capacity_priorities(self, capacity_dimensions: Dict) -> List[Dict[str, Any]]:
        """Identify priority capacity building interventions."""
        priorities = []
        
        # Sort dimensions by score
        sorted_dimensions = sorted(
            capacity_dimensions.items(),
            key=lambda x: x[1]['score']
        )
        
        for dimension, assessment in sorted_dimensions:
            if assessment['score'] < 6.0:
                priority = {
                    'dimension': dimension,
                    'current_score': assessment['score'],
                    'priority_level': 'High' if assessment['score'] < 4.0 else 'Medium',
                    'intervention_type': self._get_intervention_type(dimension),
                    'estimated_timeframe': '2-3 years',
                    'expected_improvement': 2.0 if assessment['score'] < 4.0 else 1.5
                }
                priorities.append(priority)
        
        return priorities
    
    def _get_intervention_type(self, dimension: str) -> str:
        """Get intervention type for capacity dimension."""
        interventions = {
            'technical_capacity': 'Training and skill development',
            'financial_capacity': 'Financial management and revenue enhancement',
            'human_resources': 'HR development and retention programs',
            'organizational_systems': 'Systems strengthening and process improvement',
            'leadership_governance': 'Leadership development and governance reform'
        }
        
        return interventions.get(dimension, 'General capacity building')
    
    def _analyze_institutional_landscape(self) -> Dict[str, Any]:
        """Analyze overall institutional landscape."""
        # Calculate average capacity scores by level
        national_scores = []
        local_scores = []
        
        for inst_data in self.institutional_structures['national_level'].values():
            if 'capacity_score' in inst_data:
                national_scores.append(inst_data['capacity_score'])
            elif 'average_capacity_score' in inst_data:
                national_scores.append(inst_data['average_capacity_score'])
        
        for inst_data in self.institutional_structures['local_level'].values():
            if 'capacity_score' in inst_data:
                local_scores.append(inst_data['capacity_score'])
            elif 'effectiveness_score' in inst_data:
                local_scores.append(inst_data['effectiveness_score'])
        
        # Coordination assessment
        coordination_effectiveness = self._assess_coordination_mechanisms()
        
        # Overall landscape assessment
        overall_capacity = {
            'national_level_avg': np.mean(national_scores) if national_scores else 5.0,
            'local_level_avg': np.mean(local_scores) if local_scores else 5.0,
            'coordination_effectiveness': coordination_effectiveness
        }
        
        # Identify systemic issues
        systemic_issues = self._identify_systemic_issues()
        
        # Reform recommendations
        reform_recommendations = self._generate_reform_recommendations(
            overall_capacity, systemic_issues
        )
        
        return {
            'overall_capacity': overall_capacity,
            'capacity_gaps': self._identify_capacity_gaps(),
            'coordination_assessment': coordination_effectiveness,
            'systemic_issues': systemic_issues,
            'reform_recommendations': reform_recommendations,
            'priority_reforms': self._prioritize_reforms(reform_recommendations)
        }
    
    def _assess_coordination_mechanisms(self) -> Dict[str, Any]:
        """Assess effectiveness of coordination mechanisms."""
        mechanisms = self.institutional_structures['coordination_mechanisms']
        
        effectiveness_scores = []
        for mechanism, data in mechanisms.items():
            if 'effectiveness' in data:
                effectiveness_map = {
                    'Very Low': 1, 'Low': 3, 'Medium': 5, 'High': 7, 'Very High': 9
                }
                score = effectiveness_map.get(data['effectiveness'], 3)
                effectiveness_scores.append(score)
            elif 'average_effectiveness' in data:
                effectiveness_map = {
                    'Very Low': 1, 'Low': 3, 'Medium': 5, 'High': 7, 'Very High': 9
                }
                score = effectiveness_map.get(data['average_effectiveness'], 3)
                effectiveness_scores.append(score)
        
        avg_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 3.0
        
        return {
            'average_effectiveness_score': avg_effectiveness,
            'effectiveness_rating': self._score_to_rating(avg_effectiveness),
            'key_issues': [
                'Irregular meetings',
                'Lack of decision-making authority',
                'Poor follow-up',
                'Conflicting mandates'
            ],
            'improvement_needs': [
                'Strengthen mandates',
                'Regular meeting schedules',
                'Clear accountability mechanisms',
                'Adequate resources'
            ]
        }
    
    def _identify_systemic_issues(self) -> List[Dict[str, Any]]:
        """Identify systemic issues in institutional landscape."""
        issues = [
            {
                'issue': 'Fragmented institutional mandates',
                'severity': 'High',
                'impact': 'Coordination challenges and service gaps',
                'affected_institutions': 'Multiple agencies with overlapping functions'
            },
            {
                'issue': 'Weak coordination mechanisms',
                'severity': 'High',
                'impact': 'Inefficient resource use and conflicting policies',
                'affected_institutions': 'All levels of government'
            },
            {
                'issue': 'Inadequate financial resources',
                'severity': 'High',
                'impact': 'Limited service delivery and infrastructure development',
                'affected_institutions': 'Particularly local level institutions'
            },
            {
                'issue': 'Capacity constraints',
                'severity': 'Medium',
                'impact': 'Poor implementation and service quality',
                'affected_institutions': 'Both national and local levels'
            },
            {
                'issue': 'Limited accountability mechanisms',
                'severity': 'Medium',
                'impact': 'Reduced performance and public trust',
                'affected_institutions': 'Service delivery agencies'
            }
        ]
        
        return issues
    
    def _identify_capacity_gaps(self) -> Dict[str, List[str]]:
        """Identify key capacity gaps across institutions."""
        return {
            'technical_gaps': [
                'Advanced water treatment technologies',
                'Climate change adaptation planning',
                'Integrated water resources management',
                'Water quality monitoring and assessment'
            ],
            'financial_gaps': [
                'Revenue generation and cost recovery',
                'Financial planning and budgeting',
                'Investment planning and appraisal',
                'Public-private partnership development'
            ],
            'institutional_gaps': [
                'Inter-agency coordination',
                'Performance monitoring and evaluation',
                'Strategic planning and implementation',
                'Stakeholder engagement and communication'
            ],
            'human_resource_gaps': [
                'Specialized technical skills',
                'Management and leadership capabilities',
                'Project management expertise',
                'Data analysis and information management'
            ]
        }
    
    def _generate_reform_recommendations(self,
                                       overall_capacity: Dict,
                                       systemic_issues: List[Dict]) -> List[str]:
        """Generate institutional reform recommendations."""
        recommendations = []
        
        # Based on capacity levels
        if overall_capacity['national_level_avg'] < 6.0:
            recommendations.append("Strengthen national-level institutional capacity through targeted interventions")
        
        if overall_capacity['local_level_avg'] < 5.0:
            recommendations.append("Implement comprehensive local government capacity building program")
        
        if overall_capacity['coordination_effectiveness']['average_effectiveness_score'] < 5.0:
            recommendations.append("Establish effective coordination mechanisms with clear mandates and resources")
        
        # Based on systemic issues
        high_severity_issues = [issue for issue in systemic_issues if issue['severity'] == 'High']
        
        if len(high_severity_issues) > 2:
            recommendations.append("Implement comprehensive institutional reform program to address systemic issues")
        
        # Specific recommendations
        recommendations.extend([
            "Establish river basin organizations for integrated water management",
            "Strengthen water sector financing mechanisms and cost recovery",
            "Develop comprehensive human resource development strategy",
            "Implement performance-based management systems",
            "Enhance transparency and accountability mechanisms"
        ])
        
        return recommendations
    
    def _prioritize_reforms(self, recommendations: List[str]) -> List[Dict[str, Any]]:
        """Prioritize reform recommendations."""
        # Simplified prioritization based on impact and feasibility
        priority_reforms = [
            {
                'reform': 'Establish effective coordination mechanisms',
                'priority': 'High',
                'timeframe': 'Short-term (1-2 years)',
                'impact': 'High',
                'feasibility': 'Medium',
                'estimated_cost': 'Medium ($1-5M)'
            },
            {
                'reform': 'Strengthen local government capacity',
                'priority': 'High',
                'timeframe': 'Medium-term (3-5 years)',
                'impact': 'High',
                'feasibility': 'High',
                'estimated_cost': 'High ($10-25M)'
            },
            {
                'reform': 'Establish river basin organizations',
                'priority': 'Medium',
                'timeframe': 'Medium-term (3-5 years)',
                'impact': 'High',
                'feasibility': 'Low',
                'estimated_cost': 'High ($15-30M)'
            },
            {
                'reform': 'Implement performance-based management',
                'priority': 'Medium',
                'timeframe': 'Long-term (5-10 years)',
                'impact': 'Medium',
                'feasibility': 'Medium',
                'estimated_cost': 'Medium ($2-8M)'
            }
        ]
        
        return priority_reforms