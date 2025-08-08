"""
Performance monitoring dashboard for hierarchical agent teams.
Provides comprehensive metrics, analytics, and visualization capabilities.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import random
from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Types of metrics that can be tracked."""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    TASK_THROUGHPUT = "task_throughput"
    AGENT_UTILIZATION = "agent_utilization"
    CONFIDENCE_SCORE = "confidence_score"
    ROUTING_ACCURACY = "routing_accuracy"
    ERROR_RATE = "error_rate"
    WORKLOAD_DISTRIBUTION = "workload_distribution"


@dataclass
class PerformanceMetric:
    """Represents a performance metric data point."""
    timestamp: datetime
    metric_type: MetricType
    value: float
    agent_id: Optional[str] = None
    team_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceDashboard:
    """Comprehensive performance monitoring dashboard."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetric] = []
        self.real_time_metrics: Dict[str, Any] = {}
        self.alert_thresholds = self._setup_default_thresholds()
        self.active_alerts: List[Dict[str, Any]] = []
    
    def _setup_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Setup default alert thresholds for metrics."""
        return {
            MetricType.RESPONSE_TIME.value: {"warning": 30.0, "critical": 60.0},
            MetricType.SUCCESS_RATE.value: {"warning": 0.8, "critical": 0.7},
            MetricType.ERROR_RATE.value: {"warning": 0.1, "critical": 0.2}, 
            MetricType.CONFIDENCE_SCORE.value: {"warning": 0.6, "critical": 0.4},
            MetricType.AGENT_UTILIZATION.value: {"warning": 0.9, "critical": 0.95}
        }
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a new performance metric."""
        self.metrics_history.append(metric)
        self._check_alerts(metric)
    
    def _check_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts."""
        metric_key = metric.metric_type.value
        if metric_key not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_key]
        
        alert_level = None
        if metric.value >= thresholds.get("critical", float('inf')):
            alert_level = "critical"
        elif metric.value >= thresholds.get("warning", float('inf')):
            alert_level = "warning"
        elif metric_key in ["success_rate", "confidence_score"] and metric.value <= thresholds.get("critical", 0):
            alert_level = "critical"
        elif metric_key in ["success_rate", "confidence_score"] and metric.value <= thresholds.get("warning", 0):
            alert_level = "warning"
        
        if alert_level:
            alert = {
                "timestamp": metric.timestamp,
                "level": alert_level,
                "metric_type": metric_key,
                "value": metric.value,
                "agent_id": metric.agent_id,
                "team_id": metric.team_id,
                "message": f"{metric_key} {alert_level}: {metric.value}"
            }
            self.active_alerts.append(alert)
    
    def render_dashboard(self, hierarchical_team=None):
        """Render the complete performance dashboard."""
        st.title("📊 Performance Monitoring Dashboard")
        
        # Dashboard overview
        self.render_overview_metrics(hierarchical_team)
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Real-time Metrics",
            "🔍 Agent Performance", 
            "🏢 Team Analytics",
            "🚨 Alerts & Monitoring",
            "📋 Historical Reports"
        ])
        
        with tab1:
            self.render_realtime_metrics(hierarchical_team)
        
        with tab2:
            self.render_agent_performance(hierarchical_team)
        
        with tab3:
            self.render_team_analytics(hierarchical_team)
        
        with tab4:
            self.render_alerts_monitoring()
        
        with tab5:
            self.render_historical_reports()
    
    def render_overview_metrics(self, hierarchical_team):
        """Render overview metrics at the top of dashboard."""
        st.subheader("🎯 System Overview")
        
        # Generate mock metrics if no real team provided
        if not hierarchical_team:
            metrics = self._generate_mock_overview_metrics()
        else:
            metrics = self._extract_team_overview_metrics(hierarchical_team)
        
        # Display metrics in columns
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Active Agents", 
                     metrics.get("active_agents", 0),
                     delta=metrics.get("agents_delta", 0))
        
        with col2:
            st.metric("Tasks Completed", 
                     metrics.get("tasks_completed", 0),
                     delta=metrics.get("tasks_delta", 0))
        
        with col3:
            st.metric("Success Rate", 
                     f"{metrics.get('success_rate', 0):.1%}",
                     delta=f"{metrics.get('success_delta', 0):+.1%}")
        
        with col4:
            st.metric("Avg Response Time", 
                     f"{metrics.get('avg_response_time', 0):.1f}s",
                     delta=f"{metrics.get('response_delta', 0):+.1f}s")
        
        with col5:
            st.metric("System Load", 
                     f"{metrics.get('system_load', 0):.1%}",
                     delta=f"{metrics.get('load_delta', 0):+.1%}")
        
        with col6:
            st.metric("Alerts", 
                     len(self.active_alerts),
                     delta=metrics.get("alerts_delta", 0))
    
    def render_realtime_metrics(self, hierarchical_team):
        """Render real-time metrics visualization."""
        st.subheader("📊 Real-time Performance Metrics")
        
        # Generate real-time data
        if not hierarchical_team:
            realtime_data = self._generate_mock_realtime_data()
        else:
            realtime_data = self._extract_realtime_data(hierarchical_team)
        
        # Response time chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Response Time Trend**")
            response_times = realtime_data.get("response_times", [])
            if response_times:
                df_response = pd.DataFrame(response_times)
                fig_response = px.line(df_response, x='timestamp', y='value', 
                                     color='agent_id', title="Response Time by Agent")
                fig_response.update_layout(height=300)
                st.plotly_chart(fig_response, use_container_width=True)
            else:
                st.info("No response time data available")
        
        with col2:
            st.write("**Success Rate**")
            success_rates = realtime_data.get("success_rates", [])
            if success_rates:
                df_success = pd.DataFrame(success_rates)
                fig_success = px.bar(df_success, x='agent_id', y='rate',
                                   title="Success Rate by Agent",
                                   color='rate',
                                   color_continuous_scale='RdYlGn')
                fig_success.update_layout(height=300)
                st.plotly_chart(fig_success, use_container_width=True)
            else:
                st.info("No success rate data available")
        
        # Task throughput and workload
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**Task Throughput**")
            throughput_data = realtime_data.get("throughput", [])
            if throughput_data:
                df_throughput = pd.DataFrame(throughput_data)
                fig_throughput = px.area(df_throughput, x='timestamp', y='tasks_per_minute',
                                       title="Tasks per Minute")
                fig_throughput.update_layout(height=300)
                st.plotly_chart(fig_throughput, use_container_width=True)
            else:
                st.info("No throughput data available")
        
        with col4:
            st.write("**Workload Distribution**")
            workload_data = realtime_data.get("workload", [])
            if workload_data:
                df_workload = pd.DataFrame(workload_data)
                fig_workload = px.pie(df_workload, values='workload', names='agent_id',
                                    title="Current Workload Distribution")
                fig_workload.update_layout(height=300)
                st.plotly_chart(fig_workload, use_container_width=True)
            else:
                st.info("No workload data available")
    
    def render_agent_performance(self, hierarchical_team):
        """Render individual agent performance analysis."""
        st.subheader("🤖 Agent Performance Analysis")
        
        # Agent selection
        if hierarchical_team:
            available_agents = self._get_available_agents(hierarchical_team)
        else:
            available_agents = ["web_searcher", "content_writer", "data_analyst", "code_reviewer"]
        
        selected_agent = st.selectbox("Select Agent for Detailed Analysis", available_agents)
        
        if selected_agent:
            # Agent performance metrics
            agent_metrics = self._get_agent_metrics(selected_agent, hierarchical_team)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Tasks Handled", agent_metrics.get("tasks_handled", 0))
            
            with col2:
                st.metric("Success Rate", f"{agent_metrics.get('success_rate', 0):.1%}")
            
            with col3:
                st.metric("Avg Response Time", f"{agent_metrics.get('avg_response_time', 0):.1f}s")
            
            with col4:
                st.metric("Current Load", f"{agent_metrics.get('current_load', 0):.1%}")
            
            # Detailed performance charts
            st.write("### Performance Trends")
            
            col5, col6 = st.columns(2)
            
            with col5:
                # Performance over time
                performance_history = agent_metrics.get("performance_history", [])
                if performance_history:
                    df_perf = pd.DataFrame(performance_history)
                    fig_perf = px.line(df_perf, x='timestamp', y=['response_time', 'success_rate'],
                                     title=f"{selected_agent} Performance Over Time")
                    st.plotly_chart(fig_perf, use_container_width=True)
                else:
                    st.info(f"No performance history for {selected_agent}")
            
            with col6:
                # Task type distribution
                task_types = agent_metrics.get("task_types", [])
                if task_types:
                    df_tasks = pd.DataFrame(task_types)
                    fig_tasks = px.pie(df_tasks, values='count', names='task_type',
                                     title=f"{selected_agent} Task Type Distribution")
                    st.plotly_chart(fig_tasks, use_container_width=True)
                else:
                    st.info(f"No task type data for {selected_agent}")
            
            # Agent capabilities and recommendations
            st.write("### Agent Analysis")
            
            col7, col8 = st.columns(2)
            
            with col7:
                st.write("**Capabilities:**")
                capabilities = agent_metrics.get("capabilities", [])
                for cap in capabilities:
                    st.write(f"• {cap}")
            
            with col8:
                st.write("**Recommendations:**")
                recommendations = agent_metrics.get("recommendations", [])
                for rec in recommendations:
                    st.write(f"💡 {rec}")
    
    def render_team_analytics(self, hierarchical_team):
        """Render team-level analytics."""
        st.subheader("🏢 Team Performance Analytics")
        
        # Team comparison
        if hierarchical_team:
            teams_data = self._get_teams_data(hierarchical_team)
        else:
            teams_data = self._generate_mock_teams_data()
        
        if teams_data:
            df_teams = pd.DataFrame(teams_data)
            
            # Team performance comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Team Performance Comparison**")
                fig_team_perf = px.bar(df_teams, x='team_name', y=['success_rate', 'efficiency'],
                                     title="Team Performance Metrics",
                                     barmode='group')
                st.plotly_chart(fig_team_perf, use_container_width=True)
            
            with col2:
                st.write("**Team Workload Distribution**")
                fig_team_workload = px.scatter(df_teams, x='active_agents', y='tasks_completed',
                                             size='success_rate', color='team_name',
                                             title="Team Workload vs Performance")
                st.plotly_chart(fig_team_workload, use_container_width=True)
            
            # Team details table
            st.write("### Team Details")
            st.dataframe(df_teams, use_container_width=True)
        
        # Cross-team collaboration metrics
        st.write("### Cross-team Collaboration")
        collaboration_data = self._get_collaboration_metrics(hierarchical_team)
        
        if collaboration_data:
            # Collaboration network (simplified)
            st.write("**Team Interaction Frequency**")
            col3, col4 = st.columns(2)
            
            with col3:
                interaction_matrix = collaboration_data.get("interaction_matrix", [])
                if interaction_matrix:
                    df_interactions = pd.DataFrame(interaction_matrix)
                    fig_heatmap = px.imshow(df_interactions.values,
                                          labels=dict(x="To Team", y="From Team"),
                                          x=df_interactions.columns,
                                          y=df_interactions.index,
                                          title="Team Interaction Heatmap")
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col4:
                coordination_efficiency = collaboration_data.get("coordination_efficiency", [])
                if coordination_efficiency:
                    df_coord = pd.DataFrame(coordination_efficiency)
                    fig_coord = px.line(df_coord, x='timestamp', y='efficiency',
                                      title="Coordination Efficiency Over Time")
                    st.plotly_chart(fig_coord, use_container_width=True)
    
    def render_alerts_monitoring(self):
        """Render alerts and monitoring interface."""
        st.subheader("🚨 Alerts & Monitoring")
        
        # Active alerts
        if self.active_alerts:
            st.write("### 🔥 Active Alerts")
            
            # Filter alerts by severity
            col1, col2 = st.columns([1, 3])
            
            with col1:
                severity_filter = st.selectbox("Filter by Severity", 
                                             ["All", "Critical", "Warning"])
            
            filtered_alerts = self.active_alerts
            if severity_filter != "All":
                filtered_alerts = [alert for alert in self.active_alerts 
                                 if alert["level"].lower() == severity_filter.lower()]
            
            # Display alerts
            for alert in filtered_alerts[-10:]:  # Show last 10 alerts
                alert_color = "🔴" if alert["level"] == "critical" else "🟡"
                with st.expander(f"{alert_color} {alert['message']} - {alert['timestamp'].strftime('%H:%M:%S')}"):
                    st.write(f"**Agent:** {alert.get('agent_id', 'System')}")
                    st.write(f"**Team:** {alert.get('team_id', 'N/A')}")
                    st.write(f"**Value:** {alert['value']}")
                    st.write(f"**Timestamp:** {alert['timestamp']}")
        else:
            st.success("✅ No active alerts")
        
        # Alert configuration
        st.write("### ⚙️ Alert Configuration")
        
        with st.expander("Configure Alert Thresholds"):
            for metric_type, thresholds in self.alert_thresholds.items():
                st.write(f"**{metric_type.replace('_', ' ').title()}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    warning_val = st.number_input(
                        f"Warning threshold", 
                        value=thresholds.get("warning", 0.0),
                        key=f"warning_{metric_type}"
                    )
                
                with col2:
                    critical_val = st.number_input(
                        f"Critical threshold", 
                        value=thresholds.get("critical", 0.0),
                        key=f"critical_{metric_type}"
                    )
                
                # Update thresholds
                self.alert_thresholds[metric_type]["warning"] = warning_val
                self.alert_thresholds[metric_type]["critical"] = critical_val
        
        # System health status
        st.write("### 💊 System Health")
        
        health_metrics = self._calculate_system_health()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            overall_health = health_metrics.get("overall_health", 0.85)
            health_color = "🟢" if overall_health > 0.8 else "🟡" if overall_health > 0.6 else "🔴"
            st.metric("Overall Health", f"{health_color} {overall_health:.1%}")
        
        with col2:
            availability = health_metrics.get("availability", 0.99)
            st.metric("Availability", f"{availability:.2%}")
        
        with col3:
            error_rate = health_metrics.get("error_rate", 0.02)
            st.metric("Error Rate", f"{error_rate:.2%}")
    
    def render_historical_reports(self):
        """Render historical reports and analytics."""
        st.subheader("📋 Historical Reports")
        
        # Date range selection
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Report type selection
        report_type = st.selectbox("Report Type", [
            "Performance Summary",
            "Agent Utilization",
            "Task Completion Analysis",
            "Error Analysis",
            "Routing Efficiency"
        ])
        
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                report_data = self._generate_historical_report(
                    start_date, end_date, report_type
                )
                
                if report_data:
                    st.write(f"### {report_type} Report")
                    st.write(f"**Period:** {start_date} to {end_date}")
                    
                    # Display report content based on type
                    if report_type == "Performance Summary":
                        self._render_performance_summary_report(report_data)
                    elif report_type == "Agent Utilization":
                        self._render_utilization_report(report_data)
                    elif report_type == "Task Completion Analysis":
                        self._render_task_analysis_report(report_data)
                    elif report_type == "Error Analysis":
                        self._render_error_analysis_report(report_data)
                    elif report_type == "Routing Efficiency":
                        self._render_routing_efficiency_report(report_data)
                    
                    # Download report
                    report_json = json.dumps(report_data, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Report (JSON)",
                        data=report_json,
                        file_name=f"{report_type.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No data available for the selected period")
    
    # Helper methods for generating mock data and extracting real data
    def _generate_mock_overview_metrics(self) -> Dict[str, Any]:
        """Generate mock overview metrics for demonstration."""
        return {
            "active_agents": 12,
            "agents_delta": 2,
            "tasks_completed": 1847,
            "tasks_delta": 156,
            "success_rate": 0.943,
            "success_delta": 0.025,
            "avg_response_time": 2.3,
            "response_delta": -0.4,
            "system_load": 0.67,
            "load_delta": 0.12,
            "alerts_delta": -2
        }
    
    def _generate_mock_realtime_data(self) -> Dict[str, Any]:
        """Generate mock real-time data for demonstration."""
        import random
        
        agents = ["web_searcher", "content_writer", "data_analyst", "code_reviewer"]
        
        # Generate response times
        response_times = []
        for i in range(20):
            for agent in agents:
                response_times.append({
                    "timestamp": datetime.now() - timedelta(minutes=i),
                    "agent_id": agent,
                    "value": random.uniform(1.0, 5.0)
                })
        
        # Generate success rates
        success_rates = []
        for agent in agents:
            success_rates.append({
                "agent_id": agent,
                "rate": random.uniform(0.8, 0.98)
            })
        
        # Generate throughput data
        throughput = []
        for i in range(60):
            throughput.append({
                "timestamp": datetime.now() - timedelta(minutes=i),
                "tasks_per_minute": random.randint(15, 45)
            })
        
        # Generate workload data
        workload = []
        for agent in agents:
            workload.append({
                "agent_id": agent,
                "workload": random.randint(10, 80)
            })
        
        return {
            "response_times": response_times,
            "success_rates": success_rates,
            "throughput": throughput,
            "workload": workload
        }
    
    def _get_available_agents(self, hierarchical_team) -> List[str]:
        """Get list of available agents from hierarchical team."""
        if not hierarchical_team:
            return []
        
        # This would extract actual agent names from the team
        # For now, return mock data
        return ["web_searcher", "content_writer", "data_analyst", "code_reviewer"]
    
    def _get_agent_metrics(self, agent_id: str, hierarchical_team) -> Dict[str, Any]:
        """Get detailed metrics for a specific agent."""
        import random
        
        # Mock data - in real implementation, extract from team
        return {
            "tasks_handled": random.randint(100, 500),
            "success_rate": random.uniform(0.85, 0.98),
            "avg_response_time": random.uniform(1.5, 4.0),
            "current_load": random.uniform(0.2, 0.8),
            "capabilities": ["web_search", "information_extraction", "data_processing"],
            "recommendations": [
                "Consider load balancing during peak hours",
                "Optimize response time for complex queries",
                "Monitor success rate trends"
            ],
            "performance_history": [
                {"timestamp": datetime.now() - timedelta(hours=i), 
                 "response_time": random.uniform(1.0, 5.0),
                 "success_rate": random.uniform(0.8, 1.0)}
                for i in range(24)
            ],
            "task_types": [
                {"task_type": "Search", "count": random.randint(50, 150)},
                {"task_type": "Analysis", "count": random.randint(30, 100)},
                {"task_type": "Processing", "count": random.randint(20, 80)}
            ]
        }
    
    def _generate_mock_teams_data(self) -> List[Dict[str, Any]]:
        """Generate mock team data."""
        import random
        
        teams = ["Research Team", "Development Team", "Content Team", "Support Team"]
        
        return [
            {
                "team_name": team,
                "active_agents": random.randint(2, 6),
                "tasks_completed": random.randint(100, 400),
                "success_rate": random.uniform(0.8, 0.95),
                "efficiency": random.uniform(0.7, 0.9),
                "avg_response_time": random.uniform(2.0, 6.0)
            }
            for team in teams
        ]
    
    def _calculate_system_health(self) -> Dict[str, float]:
        """Calculate overall system health metrics."""
        # Mock calculation - in real implementation, aggregate actual metrics
        return {
            "overall_health": 0.89,
            "availability": 0.992,
            "error_rate": 0.031
        }
    
    def _generate_historical_report(self, start_date, end_date, report_type) -> Dict[str, Any]:
        """Generate historical report data."""
        # Mock report data
        return {
            "report_type": report_type,
            "period": {"start": start_date, "end": end_date},
            "summary": f"Sample {report_type} report data",
            "generated_at": datetime.now(),
            "data": {"sample": "This would contain actual report data"}
        }
    
    # Report rendering methods
    def _render_performance_summary_report(self, report_data):
        """Render performance summary report."""
        st.success("Performance summary report rendered")
        st.json(report_data)
    
    def _render_utilization_report(self, report_data):
        """Render utilization report."""
        st.success("Utilization report rendered")
        st.json(report_data)
    
    def _render_task_analysis_report(self, report_data):
        """Render task analysis report.""" 
        st.success("Task analysis report rendered")
        st.json(report_data)
    
    def _render_error_analysis_report(self, report_data):
        """Render error analysis report."""
        st.success("Error analysis report rendered")
        st.json(report_data)
    
    def _render_routing_efficiency_report(self, report_data):
        """Render routing efficiency report."""
        st.success("Routing efficiency report rendered")
        st.json(report_data)

    def _get_collaboration_metrics(self, hierarchical_team) -> Dict[str, Any]:
        """Get collaboration metrics for teams."""
        if not hierarchical_team:
            return None
        
        # Generate mock collaboration data
        teams = self._get_available_teams(hierarchical_team)
        if not teams:
            return None
        
        # Create interaction matrix
        interaction_matrix = []
        team_names = [team["name"] for team in teams]
        
        for i, team1 in enumerate(team_names):
            row = []
            for j, team2 in enumerate(team_names):
                if i == j:
                    row.append(0)  # No self-interaction
                else:
                    # Generate random interaction frequency
                    interaction = random.randint(1, 10)
                    row.append(interaction)
            interaction_matrix.append(row)
        
        # Create coordination efficiency data
        coordination_efficiency = []
        timestamps = pd.date_range(start=datetime.now() - timedelta(hours=24), 
                                 end=datetime.now(), 
                                 freq='H')
        
        for timestamp in timestamps:
            efficiency = random.uniform(0.6, 0.95)
            coordination_efficiency.append({
                'timestamp': timestamp,
                'efficiency': efficiency
            })
        
        return {
            "interaction_matrix": interaction_matrix,
            "coordination_efficiency": coordination_efficiency
        }
    
    def _get_available_teams(self, hierarchical_team) -> List[Dict[str, Any]]:
        """Get available teams from hierarchical team structure."""
        if not hierarchical_team:
            return []
        
        teams = []
        try:
            # Try to extract teams from hierarchical structure
            if hasattr(hierarchical_team, 'teams'):
                for team_id, team in hierarchical_team.teams.items():
                    teams.append({
                        "id": team_id,
                        "name": getattr(team, 'name', f"Team {team_id}"),
                        "type": getattr(team, 'team_type', 'unknown')
                    })
            elif hasattr(hierarchical_team, 'supervisors'):
                for supervisor_id, supervisor in hierarchical_team.supervisors.items():
                    teams.append({
                        "id": supervisor_id,
                        "name": f"Supervisor {supervisor_id}",
                        "type": "supervisor"
                    })
        except Exception:
            # Fallback to mock data
            teams = [
                {"id": "team1", "name": "Research Team", "type": "research"},
                {"id": "team2", "name": "Development Team", "type": "development"},
                {"id": "team3", "name": "Content Team", "type": "content"},
                {"id": "team4", "name": "Support Team", "type": "support"}
            ]
        
        return teams