# Test Corrected Column Search
# Test with the correct property name to verify the fix works

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_corrected_multi_intent_search():
    """Test multi-intent search with corrected property names"""
    
    print("🧪 TESTING CORRECTED MULTI-INTENT SEARCH")
    print("=" * 50)
    
    try:
        from tools.weaviate_tools import WeaviateClientSingleton
        from weaviate.classes.query import Filter
        import weaviate.classes.query as wq
        
        client = WeaviateClientSingleton.get_instance()
        if not client:
            print("❌ No Weaviate connection")
            return
        
        column_collection = client.collections.get("Column")
        
        print("Testing the exact search concepts for your query:")
        print("'Which customers have the highest revenue and what are their names and numbers?'")
        
        # Test multi-intent search concepts
        search_concepts = [
            "high revenue customers",
            "customer names and identifiers", 
            "customer contact numbers phone email",
            "first_name last_name customer_name",
            "phone email contact",
            "revenue lifetime_value"
        ]
        
        all_found_columns = {}
        
        for i, concept in enumerate(search_concepts, 1):
            print(f"\n{i}. 🔍 Concept: '{concept}'")
            
            try:
                # Test general search (no table filter)
                response = column_collection.query.near_text(
                    query=concept,
                    limit=5,
                    return_properties=[
                        "columnName", "parentAthenaTableName", "athenaDataType", 
                        "businessName", "description", "semanticType"
                    ],
                    return_metadata=wq.MetadataQuery(distance=True)
                )
                
                if response.objects:
                    print(f"   ✅ Found {len(response.objects)} columns:")
                    for obj in response.objects:
                        props = obj.properties
                        col_name = props.get('columnName', 'Unknown')
                        table_name = props.get('parentAthenaTableName', 'Unknown')  # ✅ CORRECT PROPERTY
                        data_type = props.get('athenaDataType', 'Unknown')
                        business_name = props.get('businessName', '')
                        semantic_type = props.get('semanticType', '')
                        distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0
                        
                        # Store found columns
                        if table_name not in all_found_columns:
                            all_found_columns[table_name] = []
                        
                        all_found_columns[table_name].append({
                            'column': col_name,
                            'type': data_type,
                            'business_name': business_name,
                            'semantic_type': semantic_type,
                            'distance': distance,
                            'found_by_concept': concept
                        })
                        
                        business_display = f" ({business_name})" if business_name else ""
                        semantic_display = f" [{semantic_type}]" if semantic_type else ""
                        
                        print(f"      • {table_name}.{col_name} ({data_type}){business_display}{semantic_display}")
                        print(f"        Distance: {distance:.3f}")
                else:
                    print("   ❌ No columns found")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Summary by table
        print(f"\n📊 SUMMARY BY TABLE:")
        print("-" * 30)
        
        for table_name, columns in all_found_columns.items():
            print(f"\n🏷️  {table_name}:")
            
            # Group by intent category
            name_cols = [c for c in columns if any(term in c['found_by_concept'].lower() for term in ['name', 'identifier'])]
            contact_cols = [c for c in columns if any(term in c['found_by_concept'].lower() for term in ['contact', 'phone', 'email'])]
            revenue_cols = [c for c in columns if any(term in c['found_by_concept'].lower() for term in ['revenue', 'value'])]
            
            if name_cols:
                print("   👤 NAME COLUMNS:")
                for col in name_cols:
                    print(f"      • {col['column']} ({col['type']}) - {col['found_by_concept']}")
            
            if contact_cols:
                print("   📞 CONTACT COLUMNS:")
                for col in contact_cols:
                    print(f"      • {col['column']} ({col['type']}) - {col['found_by_concept']}")
            
            if revenue_cols:
                print("   💰 REVENUE COLUMNS:")
                for col in revenue_cols:
                    print(f"      • {col['column']} ({col['type']}) - {col['found_by_concept']}")
        
        # Test customer table specifically
        print(f"\n🎯 TESTING CUSTOMER TABLE SPECIFICALLY:")
        print("-" * 45)
        
        customer_tables = ['customer', 'customers', 'amspoc3test.customer', 'amspoc3test.customers']
        
        for table_name in customer_tables:
            print(f"\n🏷️  Testing: '{table_name}'")
            
            try:
                # ✅ CORRECT FILTER PROPERTY
                table_filter = Filter.by_property("parentAthenaTableName").equal(table_name)
                
                response = column_collection.query.fetch_objects(
                    filters=table_filter,
                    limit=10,
                    return_properties=[
                        "columnName", "athenaDataType", "businessName", 
                        "description", "semanticType", "sampleValues"
                    ]
                )
                
                if response.objects:
                    print(f"   ✅ Found {len(response.objects)} columns in {table_name}:")
                    
                    for obj in response.objects:
                        props = obj.properties
                        col_name = props.get('columnName', 'Unknown')
                        data_type = props.get('athenaDataType', 'Unknown')
                        business_name = props.get('businessName', '')
                        semantic_type = props.get('semanticType', '')
                        sample_values = props.get('sampleValues', [])
                        
                        business_display = f" ({business_name})" if business_name else ""
                        semantic_display = f" [{semantic_type}]" if semantic_type else ""
                        sample_display = f" samples: {sample_values[:3]}" if sample_values else ""
                        
                        print(f"      • {col_name} ({data_type}){business_display}{semantic_display}{sample_display}")
                    
                    break  # Found the table, no need to test other variations
                else:
                    print(f"   ❌ No columns found for '{table_name}'")
                    
            except Exception as e:
                print(f"   ❌ Error testing '{table_name}': {e}")
        
        # Final assessment
        print(f"\n🎯 MULTI-INTENT QUERY ASSESSMENT:")
        print("-" * 40)
        
        total_tables = len(all_found_columns)
        has_name_columns = any('name' in str(cols).lower() for cols in all_found_columns.values())
        has_contact_columns = any(any(term in str(cols).lower() for term in ['phone', 'email', 'contact']) for cols in all_found_columns.values())
        has_revenue_columns = any(any(term in str(cols).lower() for term in ['revenue', 'value']) for cols in all_found_columns.values())
        
        print(f"   📊 Tables found: {total_tables}")
        print(f"   👤 Name columns available: {'✅ YES' if has_name_columns else '❌ NO'}")
        print(f"   📞 Contact columns available: {'✅ YES' if has_contact_columns else '❌ NO'}")
        print(f"   💰 Revenue columns available: {'✅ YES' if has_revenue_columns else '❌ NO'}")
        
        if has_name_columns and has_contact_columns and has_revenue_columns:
            print(f"\n🎉 CONCLUSION: Your multi-intent query decomposition approach WILL WORK!")
            print("   All required column types are discoverable with corrected property names.")
        else:
            print(f"\n⚠️ CONCLUSION: Some column types may be missing.")
            print("   Check your database schema or adjust search terms.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_corrected_multi_intent_search()